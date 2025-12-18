import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.interpolate
import scipy.special
import cluster
import particlefilter_desc as particlefilter
import pynclt
import util
import poles_extractor
import argparse
import time
import collections
from kmeans_ivf import KMeansIVF
from zeroAwareIVF import ZeroAwareIVF
from SalsaNext import * 
import torch
import sys

import feature_utils
import report_utils

# --- Argument Parsing & Model Loading ---
parser = argparse.ArgumentParser(description='Pole Loc Learning with Descriptor')
parser.add_argument('--quant', nargs='?', const=6, type=int, help='Enable quant mode with specified bits (default: 6)')
parser.add_argument('--nlist', type=int, default=None, help='FAISS IVF nlist')
parser.add_argument('--nprobe', type=int, default=None, help='FAISS IVF nprobe')
parser.add_argument('--n_mapdetections', type=int, default=5, help='number of mapdetections')
parser.add_argument('--mapinterval', type=float, default=0.25, help='mapinterval')
parser.add_argument('--n_locdetections', type=int, default=1, help='n_locdetections')
parser.add_argument('--desc_dim', type=int, default=64, help='descriptor dimension (default: 64)')
parser.add_argument('--eval_out', type=str, default=None, help='path to save evaluation summary (default: stdout)')
parser.add_argument('--mode', type=str, default='full',
                    choices=['full', 'build_map', 'localize'],
                    help='full: build map + localize + evaluate; '
                        'build_map: only build global map; '
                        'localize: only localize & evaluate with existing map')
parser.add_argument('--session_start', type=int, default=0,
                    help='start index in pynclt.sessions (inclusive)')
parser.add_argument('--session_end', type=int, default=len(pynclt.sessions),
                    help='end index in pynclt.sessions (exclusive)')
args = parser.parse_args()

# Load Model
model = SalsaNext(2)
filename = './model/PoleNet'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
w_dict = torch.load(filename, map_location=lambda storage, loc: storage)
model.load_state_dict(w_dict['state_dict'], strict=True)
print('Model loaded from %s.' % filename)
model.to(device)
model.eval()

# --- Global Parameters ---
mapextent = np.array([30.0, 30.0, 5.0])
mapsize = np.full(3, 0.2)
mapshape = np.array(mapextent / mapsize, dtype=int)
mapinterval = args.mapinterval
mapdistance = mapinterval
remapdistance = 10.0
n_mapdetections = args.n_mapdetections
n_locdetections = args.n_locdetections
n_localmaps = n_mapdetections
desc_dim = args.desc_dim

T_mc_r = pynclt.T_w_o
T_r_mc = util.invert_ht(T_mc_r)
T_m_mc = np.identity(4)
T_m_mc[:3, 3] = np.hstack([0.5 * mapextent[:2], 0.5])
T_mc_m = util.invert_ht(T_m_mc)
T_m_r = T_m_mc.dot(T_mc_r)
T_r_m = util.invert_ht(T_m_r)

# --- File Naming Helpers ---
def get_globalmapname():
    return 'globalmap_{:.0f}_{:.2f}_{:.2f}_learning'.format(
        n_mapdetections, mapinterval, 0.08)

def get_locfileprefix():
    return ('localization_{:.0f}_{:.2f}_{:.2f}_{:.2f}_learning_ivf_'+str(n_locdetections)).format(
        n_mapdetections, mapinterval, 0.08, 0.20)

def get_localmapfile():
    return 'localmaps_{:.0f}_{:.2f}_{:.2f}_learning.npz'.format(
        n_mapdetections, mapinterval, 0.08)

def get_evalfile():
    return 'evaluation_{:.0f}_{:.2f}_{:.2f}_learning.npz'.format(
        n_mapdetections, mapinterval, 0.08)

def get_map_indices(session):
    distance = np.hstack([0.0, np.cumsum(np.linalg.norm(
        np.diff(session.T_w_r_gt_velo[:, :3, 3], axis=0), axis=1))])
    istart = []
    imid = []
    iend = []
    i = 0
    j = 0
    k = 0
    for id, d in enumerate(distance):
        if d >= i * mapinterval:
            istart.append(id); i += 1
        if d >= j * mapinterval + 0.5 * mapdistance:
            imid.append(id); j += 1
        if d > k * mapinterval + mapdistance:
            iend.append(id); k += 1
    return istart[:len(iend)], imid[:len(iend)], iend

def save_global_map_position():
    target_sessions = pynclt.sessions[args.session_start:args.session_end]
    globalmappos = np.empty([0, 2])
    mapfactors = np.full(len(target_sessions), np.nan)
    poleparams = np.empty([0, 3])
    all_descs = []
    for isession, s in enumerate(target_sessions):
        print(s)
        session = pynclt.session(s)
        istart, imid, iend = get_map_indices(session)
        localmappos = session.T_w_r_gt_velo[imid, :2, 3]
        if globalmappos.size == 0:
            imaps = range(localmappos.shape[0])
        else:
            imaps = []
            for imap in range(localmappos.shape[0]):
                distance = np.linalg.norm(
                    localmappos[imap] - globalmappos, axis=1).min()
                if distance > remapdistance:
                    imaps.append(imap)
        globalmappos = np.vstack([globalmappos, localmappos[imaps]])
        mapfactors[isession] = np.true_divide(len(imaps), len(imid))

        with progressbar.ProgressBar(max_value=len(imaps)) as bar:
            for iimap, imap in enumerate(imaps):
                iscan = imid[imap]
                xyz, _ = session.get_velo(iscan)
                
                 # Learning-based Detection
                localpoleparam, desc = poles_extractor.detect_poles_learning(
                    xyz, model, device, cut_z=False, desc_dim=desc_dim, desc=True, vis=False)
                # if len(desc) < 4: continue 
                all_descs.append(desc)

                # localpoleparam = poles_extractor.detect_poles(xyz)
                localpoleparam_xy = localpoleparam[:, :2]
                localpoleparam_xy = localpoleparam_xy.T
                localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
                localpoleparam_xy = np.matmul(session.T_w_r_gt_velo[imid[imap]], localpoleparam_xy)
                localpoleparam[:, :2] = localpoleparam_xy[:2,:].T
                poleparams = np.vstack([poleparams, localpoleparam])

                bar.update(iimap)

    xy = poleparams[:, :2]
    a = poleparams[:, [2]]
    boxes = np.hstack([xy - a, xy + a])
    clustermeans = np.empty([0, 3])
    clusterdescs = np.empty((0, desc_dim))
    descs_array = np.vstack(all_descs)
    print("poleparams.shape:", poleparams.shape, "descs_array.shape:", descs_array.shape)
    
    for ci in cluster.cluster_boxes(boxes):
        ci = list(ci)
        if len(ci) < n_mapdetections:
            continue
        
        # build my mean pole and descriptor
        local_poleparams = poleparams[ci, :]      # (Nc, 3)
        local_descs = descs_array[ci, :]
        
        cluster_means, cluster_descs = feature_utils.merge_cluster_descriptors(
            local_poleparams,
            local_descs,
            threshold=0.2
        )
        clustermeans = np.vstack([clustermeans, cluster_means])
        clusterdescs = np.vstack([clusterdescs, cluster_descs])

        
    globalmapfile = os.path.join('nclt', get_globalmapname() + '.npz')
    np.savez(globalmapfile,
                polemeans=clustermeans,
                descmeans=clusterdescs,
                mapfactors=mapfactors,
                mappos=globalmappos,
                allpole=poleparams,
                descriptors=descs_array)
    
    print(f"allpole.shape: {poleparams.shape}, descriptors.shape: {descs_array.shape}")
    print(f"polemeans.shape: {clustermeans.shape}, descmeans.shape: {clusterdescs.shape}")
    report_utils.plot_global_map(globalmapfile)


def save_global_map():
    target_sessions = pynclt.sessions[args.session_start:args.session_end]
    POS_MATCH_THRESHOLD_METERS = 8.0
    DESC_SCORE_TOLERANCE = 0.2
    DESC_MIN_SCORE_THRESHOLD = 3

    global_clustermeans = np.empty([0, 3])
    global_clusterdescs = np.empty((0, desc_dim))
    global_mappos = np.empty([0, 2])
    mapfactors = np.full(len(pynclt.sessions), np.nan)
    all_raw_poles_list = []
    all_raw_descs_list = []

    print(target_sessions)
    for isession, s in enumerate(target_sessions):
        print(s)
        session = pynclt.session(s)
        istart, imid, iend = get_map_indices(session)
        session_features_detected = 0
        session_features_added = 0
        session_mappos_contributed = []
        imaps = range(len(imid))

        with progressbar.ProgressBar(max_value=len(imaps)) as bar:
            for iimap in imaps:
                iscan = imid[iimap]
                xyz, _ = session.get_velo(iscan)
                
                # Learning-based Detection
                localpoleparam, desc = poles_extractor.detect_poles_learning(
                    xyz, model, device, cut_z=False, desc_dim=desc_dim, desc=True, vis=False)
                
                if localpoleparam.shape[0] == 0:
                    bar.update(iimap); continue

                localpoleparam_xy_local = localpoleparam[:, :2].T
                localpoleparam_xy_local = np.vstack([
                    localpoleparam_xy_local, 
                    np.zeros_like(localpoleparam_xy_local[0]), 
                    np.ones_like(localpoleparam_xy_local[0])
                ])
                T_w_r = session.T_w_r_gt_velo[imid[iimap]]
                localpoleparam_xy_global = np.matmul(T_w_r, localpoleparam_xy_local)
                localpoleparam_global = localpoleparam.copy()
                localpoleparam_global[:, :2] = localpoleparam_xy_global[:2, :].T

                all_raw_poles_list.append(localpoleparam_global)
                all_raw_descs_list.append(desc)

                # Local Clustering
                xy = localpoleparam_global[:, :2]
                a = localpoleparam_global[:, [2]]
                boxes = np.hstack([xy - a, xy + a])
                
                for ci in cluster.cluster_boxes(boxes):
                    ci = list(ci)
                    if not ci: continue
                    session_features_detected += 1
                    
                    merged_poles, merged_descs = feature_utils.merge_cluster_descriptors(
                        localpoleparam_global[ci, :], desc[ci, :], threshold=0.2
                    )
                    if merged_poles.shape[0] == 0: continue
                        
                    new_pole = np.average(merged_poles, axis=0)
                    new_desc = np.average(merged_descs, axis=0)
                    
                    match_idx = feature_utils.score_based_feature_match(
                        new_pole, new_desc,
                        global_clustermeans, global_clusterdescs,
                        POS_MATCH_THRESHOLD_METERS,
                        DESC_MIN_SCORE_THRESHOLD,
                        DESC_SCORE_TOLERANCE
                    )
                    
                    if match_idx == -1:
                        session_features_added += 1
                        global_clustermeans = np.vstack([global_clustermeans, new_pole])
                        global_clusterdescs = np.vstack([global_clusterdescs, new_desc])
                        current_pos = session.T_w_r_gt_velo[imid[iimap], :2, 3]
                        session_mappos_contributed.append(current_pos)
                bar.update(iimap)
        
        if session_features_detected > 0:
            mapfactors[isession] = np.true_divide(session_features_added, session_features_detected)
        else:
            mapfactors[isession] = 0.0
        if session_mappos_contributed:
            global_mappos = np.vstack([global_mappos, np.unique(np.array(session_mappos_contributed), axis=0)])

    if all_raw_poles_list:
        all_raw_poles_for_npz = np.vstack(all_raw_poles_list)
        all_raw_descs_for_npz = np.vstack(all_raw_descs_list)
    else:
        all_raw_poles_for_npz = np.empty([0, 3])
        all_raw_descs_for_npz = np.empty((0, desc_dim))

    if global_mappos.shape[0] > 0:
        global_mappos = np.unique(global_mappos, axis=0)
        
    globalmapfile = os.path.join('nclt', get_globalmapname() + '.npz')
    np.savez(globalmapfile,
             polemeans=global_clustermeans,
             descmeans=global_clusterdescs,
             mapfactors=mapfactors,
             mappos=global_mappos,
             allpole=all_raw_poles_for_npz,
             descriptors=all_raw_descs_for_npz)
    
    print(f"--- Incremental Map Build Complete ---")
    report_utils.plot_global_map(globalmapfile)

def save_local_maps(sessionname, visualize=False):
    print(sessionname)
    session = pynclt.session(sessionname)
    util.makedirs(session.dir)
    istart, imid, iend = get_map_indices(session)
    maps = []
    all_descs = []
    
    detection_times_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def plot_timing_analysis(times, save_dir):
        plt.figure(figsize=(10, 6))
        plt.plot(times, linestyle='-', color='b', linewidth=1, alpha=0.8)
        plt.title(f"Detection Latency Analysis ({sessionname})")
        plt.xlabel("Frame Sequence Index")
        plt.ylabel("Inference Time (ms)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        avg_time = np.mean(times)
        plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Avg: {avg_time:.2f} ms')
        plt.legend()

        output_path = os.path.join(save_dir, "detection_timing.png")
        plt.savefig(output_path)
        plt.close() 
        print(f"Timing plot saved to: {output_path}")

    with progressbar.ProgressBar(max_value=len(iend)) as bar:
        for i in range(len(iend)):
            # Pre-processing
            T_w_mc = util.project_xy(session.T_w_r_odo_velo[imid[i]].dot(T_r_mc))
            T_w_m = T_w_mc.dot(T_mc_m)
            T_m_w = util.invert_ht(T_w_m)
            T_w_r = session.T_w_r_odo_velo[imid[i]]
            T_m_r = np.matmul(T_m_w, T_w_r)
            T_g_mc = util.project_xy(session.T_w_r_gt_velo[imid[i]].dot(T_r_mc))
            T_g_m = T_g_mc.dot(T_mc_m)

            iscan = imid[i]
            xyz, _ = session.get_velo(iscan)
            
            t0 = time.perf_counter_ns()
            poleparams, desc = poles_extractor.detect_poles_learning(
                xyz, model, device, cut_z=False, desc_dim=desc_dim, desc=True, vis=False)
            all_descs.append(desc)
                

            localpoleparam_xy = poleparams[:, :2]
            localpoleparam_xy = localpoleparam_xy.T
            localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])])
            localpoleparam_xy = np.matmul(T_m_r, localpoleparam_xy)
            poleparams[:, :2] = localpoleparam_xy[:2,:].T

            map = {'poleparams': poleparams, 'T_w_m': T_w_m, 'T_g_m': T_g_m,
                'istart': istart[i], 'imid': imid[i], 'iend': iend[i]}
            map['descriptors'] = desc
            maps.append(map)
            bar.update(i)
            
    if detection_times_ms:
        plot_timing_analysis(detection_times_ms, session.dir)
    
    np.savez(os.path.join(session.dir, get_localmapfile()), maps=maps)

def localize(sessionname, visualize=False, quant=False, quant_bits=None, ivf_nlist=None, ivf_nprobe=None):
    print(sessionname)
    print(f"[localize] quant={quant}")
    append_count = 0
    
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans'][:, :2]
    descmap = mapdata['descmeans']
    descxy = mapdata['polemeans'][:, :2]
    
    qmap, thresholds = [], []
    if quant:
        qmap, thresholds = feature_utils.quantize_descmap(descmap, bits=quant_bits)
    
    print(f"descmap size: {descmap.shape}, descxy size: {descxy.shape}")
    if quant: print([f"{t:.10f}" for t in thresholds])
    
    polevar = 1.50
    session = pynclt.session(sessionname)
    locdata = np.load(os.path.join(session.dir, get_localmapfile()), allow_pickle=True)['maps']
    polepos_m = []
    polepos_w = []
    desclocal = []
    meas_events = []
    for i in range(len(locdata)):
        n = locdata[i]['poleparams'].shape[0]
        pad = np.hstack([np.zeros([n, 1]), np.ones([n, 1])])
        polepos_m.append(np.hstack([locdata[i]['poleparams'][:, :2], pad]).T)
        polepos_w.append(locdata[i]['T_w_m'].dot(polepos_m[i]))
        desclocal.append(locdata[i]['descriptors'])
        
    istart = 0
    T_w_r_start = util.project_xy(session.get_T_w_r_gt(session.t_relodo[istart]).dot(T_r_mc)).dot(T_mc_r)
    
    desc_source = qmap if quant else descmap
    map_ids = np.arange(desc_source.shape[0], dtype=np.int64)
    ivf = ZeroAwareIVF()
    if ivf_nlist is not None:
        if ivf_nlist < 1: raise ValueError("--nlist must be >= 1")
        ivf.nlist = ivf_nlist
    if ivf_nprobe is not None:
        if ivf_nprobe < 1: raise ValueError("--nprobe must be >= 1")
        ivf.nprobe = ivf_nprobe        
    build_stats = ivf.build(desc_source, map_ids=map_ids)
    print("[IVF] build stats:", build_stats)
    descmap_index, edges = ivf, None
    
    filter = particlefilter.particlefilter(10000, 
        T_w_r_start, 2.5, np.radians(5.0), polemap, polevar, qmap if quant else descmap, descxy, descmap_index, edges, quant, T_w_o=T_mc_r)
    filter.estimatetype = 'best'
    filter.minneff = 0.5
    
    knn_filter = particlefilter.particlefilter(500, 
        T_w_r_start, 2.5, np.radians(5.0), polemap, polevar, qmap if quant else descmap, descxy, descmap_index, edges, quant, T_w_o=T_mc_r)
    knn_filter.estimatetype = 'best'
    knn_filter.minneff = 0.5


    imap = 0
    while imap < locdata.shape[0] - 1 and \
            session.t_velo[locdata[imap]['iend']] < session.t_relodo[istart]:
        imap += 1
    
    T_w_r_est = np.full([session.t_relodo.size, 4, 4], np.nan)
    T_w_r_est_knn = np.full([session.t_relodo.size, 4, 4], np.nan)
    T = session.t_relodo.size
    tim_total_ms            = np.zeros(T, dtype=np.float32) 
    tim_update_measurement_ms = np.zeros(T, dtype=np.float32) 
    tim_estimate_pose_ms    = np.zeros(T, dtype=np.float32)
    tim_knn_update_ms       = np.zeros(T, dtype=np.float32)
    tim_knn_measurement_ms  = np.zeros(T, dtype=np.float32)
    tim_knn_estimate_ms     = np.zeros(T, dtype=np.float32)
    n_active_per_step         = np.zeros(T, dtype=np.int32)

    with progressbar.ProgressBar(max_value=session.t_relodo.size) as bar:
        for i in range(istart, session.t_relodo.size):
            t_step0_ns = time.perf_counter_ns()
            relodocov = np.empty([3, 3])
            relodocov[:2, :2] = session.relodocov[i, :2, :2]
            relodocov[:, 2] = session.relodocov[i, [0, 1, 5], 5]
            relodocov[2, :] = session.relodocov[i, 5, [0, 1, 5]]
            filter.update_motion(session.relodo[i], relodocov * 2.0**2)
            
            t0_ns = time.perf_counter_ns()
            T_w_r_est[i] = filter.estimate_pose()
            tim_estimate_pose_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6
            
            # KNN motion
            knn_filter.update_motion(session.relodo[i], relodocov * 2.0**2)
            t0_ns = time.perf_counter_ns()
            T_w_r_est_knn[i] = knn_filter.estimate_pose()
            tim_knn_estimate_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6
            
            t_now = session.t_relodo[i]
            if imap < locdata.shape[0]:
                t_end = session.t_velo[locdata[imap]['iend']]
                if t_now >= t_end:
                    imaps = range(imap, np.clip(imap-n_localmaps, -1, None), -1)
                    xy = np.hstack([polepos_w[j][:2] for j in imaps]).T
                    a = np.vstack([ld['poleparams'][:, [2]] for ld in locdata[imaps]])
                    boxes = np.hstack([xy - a, xy + a])
                    ipoles = set(range(polepos_w[imap].shape[1]))
                    iactive = set()
                    for ci in cluster.cluster_boxes(boxes):
                        if len(ci) >= n_locdetections:
                            iactive |= set(ipoles) & ci
                    iactive = list(iactive)
                    
                    meas_events.append({'t_now': float(t_now), 'n_active': int(len(iactive))})
                    append_count += 1
                    
                    # desc
                    if len(iactive) >= 4:
                        n_active_per_step[i] = len(iactive)
                        t_mid = session.t_velo[locdata[imap]['imid']]
                        T_w_r_mid = util.project_xy(session.get_T_w_r_odo(t_mid).dot(T_r_mc)).dot(T_mc_r)
                        T_w_r_now = util.project_xy(session.get_T_w_r_odo(t_now).dot(T_r_mc)).dot(T_mc_r)
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(T_r_m).dot(polepos_m[imap][:, iactive])
                        
                        desc = desclocal[imap][iactive]
                        if quant:
                            desc = feature_utils.quantize_descriptor(desc, thresholds)

                        t0_ns = time.perf_counter_ns()
                        filter.update_measurement(desc, polepos_r_now[:2].T)
                        tim_update_measurement_ms[i] = (time.perf_counter_ns() - t0_ns) / 1e6
                        
                        t0_ns = time.perf_counter_ns()
                        T_w_r_est[i] = filter.estimate_pose()
                        tim_estimate_pose_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6
                    
                    # knn
                    if iactive:
                        t_mid = session.t_velo[locdata[imap]['imid']]
                        T_w_r_mid = util.project_xy(session.get_T_w_r_odo(t_mid).dot(T_r_mc)).dot(T_mc_r)
                        T_w_r_now = util.project_xy(session.get_T_w_r_odo(t_now).dot(T_r_mc)).dot(T_mc_r)
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(T_r_m).dot(polepos_m[imap][:, iactive])
                        
                        t0_ns = time.perf_counter_ns()
                        knn_filter.update_measurement_knn(polepos_r_now[:2].T)
                        tim_knn_measurement_ms[i] = (time.perf_counter_ns() - t0_ns) / 1e6
                        
                        t0_ns = time.perf_counter_ns()
                        T_w_r_est_knn[i] = knn_filter.estimate_pose()
                        tim_knn_estimate_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6

                    imap += 1
            tim_total_ms[i] = (time.perf_counter_ns() - t_step0_ns) / 1e6
            bar.update(i)

    filename = os.path.join(session.dir, get_locfileprefix() \
        + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.npz'))
    np.savez(filename, T_w_r_est=T_w_r_est, T_w_r_est_knn=T_w_r_est_knn, meas_events=np.array(meas_events, dtype=object),
            tim_total_ms=tim_total_ms, tim_update_measurement_ms=tim_update_measurement_ms, tim_estimate_pose_ms=tim_estimate_pose_ms,
            tim_knn_update_ms=tim_knn_update_ms, tim_knn_measurement_ms=tim_knn_measurement_ms, tim_knn_estimate_ms=tim_knn_estimate_ms,
            n_active_per_step=n_active_per_step,)
    print('append_count', append_count)

def evaluate(output_path=args.eval_out):
    stats = []
    target_sessions = pynclt.sessions[args.session_start:args.session_end]
    for sessionname in target_sessions:
        files = [file for file in os.listdir(os.path.join(pynclt.resultdir, sessionname)) if file.startswith(get_locfileprefix())]
        files.sort()
        session = pynclt.session(sessionname)

        cumdist = np.hstack([0.0, np.cumsum(np.linalg.norm(np.diff(session.T_w_r_gt[:, :3, 3], axis=0), axis=1))])
        t_eval = scipy.interpolate.interp1d(cumdist, session.t_gt)(np.arange(0.0, cumdist[-1], 1.0))
        T_w_r_gt = np.stack([util.project_xy(session.get_T_w_r_gt(t).dot(T_r_mc)).dot(T_mc_r) for t in t_eval])

        T_gt_est, T_gt_est_knn = [], []
        for file in files:
            fpath = os.path.join(pynclt.resultdir, sessionname, file)
            data = np.load(fpath)
            T_w_r_est = data['T_w_r_est']
            T_w_r_est_knn = data['T_w_r_est_knn']
            
            T_w_r_est_interp = np.empty([len(t_eval), 4, 4])
            T_w_r_est_knn_interp = np.empty([len(t_eval), 4, 4])
            iodo = 1; inum = 0
            for ieval in range(len(t_eval)):
                while session.t_relodo[iodo] < t_eval[ieval]:
                    iodo += 1
                    if iodo >= session.t_relodo.shape[0]: break
                if iodo >= session.t_relodo.shape[0]: break
                T_w_r_est_interp[ieval] = util.interpolate_ht(T_w_r_est[iodo-1:iodo+1], session.t_relodo[iodo-1:iodo+1], t_eval[ieval])
                T_w_r_est_knn_interp[ieval] = util.interpolate_ht(T_w_r_est_knn[iodo-1:iodo+1], session.t_relodo[iodo-1:iodo+1], t_eval[ieval])
                inum += 1
            T_gt_est.append(np.matmul(util.invert_ht(T_w_r_gt), T_w_r_est_interp)[:inum, ...])
            T_gt_est_knn.append(np.matmul(util.invert_ht(T_w_r_gt), T_w_r_est_knn_interp)[:inum, ...])
            
        T_gt_est = np.stack(T_gt_est) 
        T_gt_est_knn = np.stack(T_gt_est_knn) 
        L = T_gt_est.shape[1]
        t_plot = t_eval[:L]  

        poserrors = np.linalg.norm(T_gt_est[..., :2, 3], axis=-1)
        poserror_mean = np.mean(poserrors, axis=0)
        poserrors_knn = np.linalg.norm(T_gt_est_knn[..., :2, 3], axis=-1)
        poserror_mean_knn = np.mean(poserrors_knn, axis=0)

        # [Refactor] Use report_utils
        report_utils.plot_evaluation_result(sessionname=sessionname, 
                               poserror_mean=poserror_mean, 
                               poserror_mean_knn=poserror_mean_knn,
                               t_plot=t_plot, 
                               files=files,
                               result_dir=pynclt.resultdir)

        lonerror = np.mean(np.mean(np.abs(T_gt_est[..., 0, 3]), axis=-1))
        laterror = np.mean(np.mean(np.abs(T_gt_est[..., 1, 3]), axis=-1))
        poserror = np.mean(np.mean(poserrors, axis=-1))
        poserror_knn = np.mean(np.mean(poserrors_knn, axis=-1))
        print(poserror, poserror_knn)
        posrmse = np.mean(np.sqrt(np.mean(poserrors**2, axis=-1)))
        angerrors = np.degrees(np.abs(np.array([util.ht2xyp(T)[:, 2] for T in T_gt_est])))
        angerror = np.mean(np.mean(angerrors, axis=-1))
        angrmse = np.mean(np.sqrt(np.mean(angerrors**2, axis=-1)))

        stats.append({'session': sessionname, 'lonerror': lonerror, 
            'laterror': laterror, 'poserror': poserror, 'posrmse': posrmse,
            'angerror': angerror, 'angrmse': angrmse, 'T_gt_est': T_gt_est})

    np.savez(os.path.join(pynclt.resultdir, get_evalfile()), stats=stats)

    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    if output_path is None:
        out = sys.stdout
        close_out = False
    else:
        out = open(output_path, 'a')
        close_out = True
    print('session \t f\te_pos \trmse_pos \te_ang \te_rmse', file=out)
    row = '{session} \t{f} \t{poserror} \t{posrmse} \t{angerror} \t{angrmse}'
    for i, stat in enumerate(stats):
        print(row.format(
            session=stat['session'],
            f=mapdata['mapfactors'][i] * 100.0,
            poserror=stat['poserror'],
            posrmse=stat['posrmse'],
            angerror=stat['angerror'],
            angrmse=stat['angrmse']),
            file=out)
        
    if close_out:
        out.close()

if __name__ == '__main__':
    if args.quant is None:
        do_quant = False
        quant_bits = None
    else:
        do_quant = True
        quant_bits = args.quant
        
    print(f"Quantization enabled? {do_quant}, bits={quant_bits}")
    ivf_nlist = args.nlist
    ivf_nprobe = args.nprobe
    
    target_sessions = pynclt.sessions[args.session_start:args.session_end]
    if args.mode == 'full':
        print(f"[MODE full] Build global map + save localmaps + localize, "
              f"sessions[{args.session_start}:{args.session_end}]")
        #save_global_map_position()
        for session in target_sessions:
            save_local_maps(session)
            localize(session, visualize=False, quant=do_quant, quant_bits=quant_bits, ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe)
        evaluate(output_path=args.eval_out)
    elif args.mode == 'build_map':
        print(f"[MODE build_map] Only build global map using "
              f"sessions[{args.session_start}:{args.session_end}]")
        # save_global_map_position()
        save_global_map()
    elif args.mode == 'localize':
        print(f"[MODE localize] Only localize & evaluate on "
              f"sessions[{args.session_start}:{args.session_end}] "
              "(assuming localmaps already exist)")
        for session in target_sessions:
            localize(session, visualize=False, quant=do_quant, quant_bits=quant_bits, ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe)
        evaluate(output_path=args.eval_out)