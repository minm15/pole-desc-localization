import datetime
import os, glob
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.interpolate
import scipy.special
import cluster
import particlefilter
import pynclt
import util
import poles_extractor
import argparse
import time
from kmeans_ivf import KMeansIVF
from zeroAwareIVF import ZeroAwareIVF

mapextent = np.array([30.0, 30.0, 5.0])
mapsize = np.full(3, 0.2)
mapshape = np.array(mapextent / mapsize, dtype=np.int)
mapinterval = 0.25
mapdistance = 0.25
remapdistance = 10.0
n_mapdetections = 6
n_locdetections = 2
n_localmaps = 6

T_mc_r = pynclt.T_w_o
T_r_mc = util.invert_ht(T_mc_r)
T_m_mc = np.identity(4)
T_m_mc[:3, 3] = np.hstack([0.5 * mapextent[:2], 0.5])
T_mc_m = util.invert_ht(T_m_mc)
T_m_r = T_m_mc.dot(T_mc_r)
T_r_m = util.invert_ht(T_m_r)

def get_globalmapname():
    return 'globalmap_{:.0f}_{:.2f}_{:.2f}'.format(
        n_mapdetections, mapinterval, 0.08)


def get_locfileprefix():
    return 'localization_{:.0f}_{:.2f}_{:.2f}_{:.2f}'.format(
        n_mapdetections, mapinterval, 0.08, 0.20)


def get_localmapfile():
    return 'localmaps_{:.0f}_{:.2f}_{:.2f}.npz'.format(
        n_mapdetections, mapinterval, 0.08)


def get_evalfile():
    return 'evaluation_{:.0f}_{:.2f}_{:.2f}.npz'.format(
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
            istart.append(id)
            i += 1
        if d >= j * mapinterval + 0.5 * mapdistance:
            imid.append(id)
            j += 1
        if d > k * mapinterval + mapdistance:
            iend.append(id)
            k += 1
    return istart[:len(iend)], imid[:len(iend)], iend


def save_global_map(use_desc=False):
    globalmappos = np.empty([0, 2])
    mapfactors = np.full(len(pynclt.sessions), np.nan)
    poleparams = np.empty([0, 3])
    all_descs = []
    for isession, s in enumerate(pynclt.sessions):
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
                
                if use_desc:
                    localpoleparam, desc = poles_extractor.detect_poles(
                        xyz, desc=True)
                    # if len(desc) < 4: continue 
                    all_descs.append(desc)
                else:
                    localpoleparam = poles_extractor.detect_poles(xyz)

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
    clusterdescs = np.empty((0,64))
    descs_array = np.vstack(all_descs)
    print(poleparams.shape, descs_array.shape)
    
    desc_txt_path = os.path.join(session.dir, 'cluster_descriptors.txt')
    cid = 0
    with open(desc_txt_path, 'w') as f_desc:
    
        for ci in cluster.cluster_boxes(boxes):
            ci = list(ci)
            if len(ci) < n_mapdetections:
                continue
            
            # build my mean pole and descriptor
            local_poleparams = poleparams[ci, :]      # (Nc, 3)
            local_descs = descs_array[ci, :]
            
            cluster_means, cluster_descs = merge_cluster_descriptors(
                local_poleparams,
                local_descs,
                threshold=0.2
            )
            clustermeans = np.vstack([clustermeans, cluster_means])
            clusterdescs = np.vstack([clusterdescs, cluster_descs])
            
            # clustermeans = np.vstack([clustermeans, np.average(poleparams[ci, :], axis=0)])
            # cluster_descs = np.vstack([cluster_descs, np.mean(descs_array[ci, :], axis=0)])
            
            f_desc.write(f"c{cid}\n")
            for i in range(len(cluster_descs)):
                vec = cluster_descs[i]
                line = " ".join(f"{v:.2f}" for v in vec)
                f_desc.write(line + "\n")

            cid += 1
        
    globalmapfile = os.path.join('nclt', get_globalmapname() + '.npz')
    allglobalmapfile = os.path.join('nclt', get_globalmapname() + '_all' + '.npz')
    if use_desc:
        all_descs = np.vstack(all_descs)
        np.savez(globalmapfile,
                 polemeans=clustermeans,
                 descmeans=clusterdescs,
                 mapfactors=mapfactors,
                 mappos=globalmappos,
                 allpole=poleparams,
                 descriptors=descs_array)
        
        # np.savez(allglobalmapfile, polemeans=xy, descriptors=descs_array)
        print(f"allpole.shape: {poleparams.shape}, descriptors.shape: {descs_array.shape}")
        print(f"polemeans.shape: {clustermeans.shape}, descmeans.shape: {clusterdescs.shape}")
        
    else:
        np.savez(globalmapfile,
                 polemeans=clustermeans,
                 mapfactors=mapfactors,
                 mappos=globalmappos)
    plot_global_map(globalmapfile)


def plot_global_map(globalmapfile):
    data = np.load(globalmapfile)
    x, y = data['polemeans'][:, :2].T
    plt.clf()
    plt.scatter(x, y, s=1, c='b', marker='.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(globalmapfile[:-4] + '.svg')
    plt.savefig(globalmapfile[:-4] + '.pgf')
    print(data['mapfactors'])


def save_local_maps(sessionname, visualize=False, use_desc=False):
    print(sessionname)
    session = pynclt.session(sessionname)
    util.makedirs(session.dir)
    istart, imid, iend = get_map_indices(session)
    maps = []
    all_descs = []
    tim_detect_ms = []
    ts_localmaps = []
    with progressbar.ProgressBar(max_value=len(iend)) as bar:
        for i in range(len(iend)):
            T_w_mc = util.project_xy(
                session.T_w_r_odo_velo[imid[i]].dot(T_r_mc))
            T_w_m = T_w_mc.dot(T_mc_m)
            T_m_w = util.invert_ht(T_w_m)
            T_w_r = session.T_w_r_odo_velo[imid[i]]
            T_m_r = np.matmul(T_m_w, T_w_r)
            T_g_mc = util.project_xy(session.T_w_r_gt_velo[imid[i]].dot(T_r_mc))
            T_g_m = T_g_mc.dot(T_mc_m)

            iscan = imid[i]
            xyz, _ = session.get_velo(iscan)
            
            t0 = time.perf_counter_ns()
            if use_desc:
                poleparams, desc = poles_extractor.detect_poles(
                    xyz, desc=True)
                all_descs.append(desc)
            else:
                poleparams = poles_extractor.detect_poles(xyz)
            tim_detect_ms.append((time.perf_counter_ns() - t0) / 1e6)
            ts_localmaps.append(session.t_velo[imid[i]])

            # poleparams = poles_extractor.detect_poles(xyz)
            localpoleparam_xy = poleparams[:, :2]
            localpoleparam_xy = localpoleparam_xy.T
            localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
            localpoleparam_xy = np.matmul(T_m_r, localpoleparam_xy)
            poleparams[:, :2] = localpoleparam_xy[:2,:].T

            map = {'poleparams': poleparams, 'T_w_m': T_w_m, 'T_g_m': T_g_m,
                'istart': istart[i], 'imid': imid[i], 'iend': iend[i]}
            if use_desc:
                map['descriptors'] = desc   # (Ni, 180) array
            maps.append(map)
            bar.update(i)
            
    np.savez(os.path.join(session.dir, get_localmapfile()), maps=maps,
             tim_detect_ms=np.array(tim_detect_ms, dtype=np.float32),
             ts_localmaps=np.asarray(ts_localmaps, dtype=np.float64))

def localize(sessionname, visualize=False, quant=False, quant_bits=None, ivf_nlist=None, ivf_nprobe=None):
    print(sessionname)
    print(f"[localize] quant={quant}")
    append_count = 0
    
    # load the global map data
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans'][:, :2]
    descmap = mapdata['descmeans']
    descxy = mapdata['polemeans'][:, :2]
    
    qmap, thresholds = [], []
    if quant:
        qmap, thresholds = quantize_descmap(descmap, bits=quant_bits)
    
    print(f"descmap size: {descmap.shape}, descxy size: {descxy.shape}")
    print([f"{t:.10f}" for t in thresholds])
    if quant:
        print(f"qmap size: {qmap.shape}")
    polevar = 1.50
    
    # load the local map data
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
    T_w_r_start = util.project_xy(
        session.get_T_w_r_gt(session.t_relodo[istart]).dot(T_r_mc)).dot(T_mc_r)
    
    # construct the descmap index
    desc_source = qmap if quant else descmap                    # shape (N,64) uint8
    map_ids = np.arange(desc_source.shape[0], dtype=np.int64)
    #ivf = KMeansIVF()
    ivf = ZeroAwareIVF()
    # read ivf argument   
    if ivf_nlist is not None:
        if ivf_nlist < 1:
            raise ValueError("--nlist must be >= 1")
        ivf.nlist = ivf_nlist
    if ivf_nprobe is not None:
        if ivf_nprobe < 1:
            raise ValueError("--nprobe must be >= 1")
        ivf.nprobe = ivf_nprobe        
    build_stats = ivf.build(desc_source, map_ids=map_ids)
    print("[IVF] build stats:", build_stats)
    #descmap_index, edges = build_descmap_index(qmap if quant else descmap, 4)
    descmap_index, edges = ivf, None
    
    ### descriptor filter ###
    filter = particlefilter.particlefilter(5000, 
        T_w_r_start, 2.5, np.radians(5.0), polemap, polevar, qmap if quant else descmap, descxy, descmap_index, edges, quant, T_w_o=T_mc_r)
    filter.estimatetype = 'best'
    filter.minneff = 0.5
    
    ### knn filter ###
    knn_filter = particlefilter.particlefilter(500, 
        T_w_r_start, 2.5, np.radians(5.0), polemap, polevar, qmap if quant else descmap, descxy, descmap_index, edges, quant, T_w_o=T_mc_r)
    knn_filter.estimatetype = 'best'
    knn_filter.minneff = 0.5

    if visualize:
        plt.ion()
        figure = plt.figure()
        nplots = 1
        mapaxes = figure.add_subplot(nplots, 1, 1)
        mapaxes.set_aspect('equal')
        mapaxes.scatter(polemap[:, 0], polemap[:, 1], s=5, c='b', marker='s')
        x_gt, y_gt = session.T_w_r_gt[::20, :2, 3].T
        mapaxes.plot(x_gt, y_gt, 'g')
        particles = mapaxes.scatter([], [], s=1, c='r')
        arrow = mapaxes.arrow(0.0, 0.0, 1.0, 0.0, length_includes_head=True, 
            head_width=0.7, head_length=1.0, color='k')
        arrowdata = np.hstack(
            [arrow.get_xy(), np.zeros([8, 1]), np.ones([8, 1])]).T
        locpoles = mapaxes.scatter([], [], s=30, c='k', marker='x')
        viewoffset = 25.0

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
            
            ### knn filter ###
            t0_ns = time.perf_counter_ns()
            knn_filter.update_motion(session.relodo[i], relodocov * 2.0**2)
            tim_knn_update_ms[i] = (time.perf_counter_ns() - t0_ns) / 1e6

            t0_ns = time.perf_counter_ns()
            T_w_r_est_knn[i] = knn_filter.estimate_pose()
            tim_knn_estimate_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6
            
            t_now = session.t_relodo[i]
            if imap < locdata.shape[0]:
                t_end = session.t_velo[locdata[imap]['iend']]
                if t_now >= t_end:
                    imaps = range(imap, np.clip(imap-n_localmaps, -1, None), -1)
                    xy = np.hstack([polepos_w[j][:2] for j in imaps]).T
                    a = np.vstack([ld['poleparams'][:, [2]] \
                        for ld in locdata[imaps]])
                    boxes = np.hstack([xy - a, xy + a])
                    ipoles = set(range(polepos_w[imap].shape[1]))
                    # iactive = ipoles
                    iactive = set()
                    for ci in cluster.cluster_boxes(boxes):
                        if len(ci) >= n_locdetections:
                            iactive |= set(ipoles) & ci
                    iactive = list(iactive)
                    
                    meas_events.append({
                        't_now': float(t_now), 
                        'n_active': int(len(iactive))
                    })
                    append_count += 1
                    
                    if len(iactive) >= 2:
                        n_active_per_step[i] = len(iactive)
                        t_mid = session.t_velo[locdata[imap]['imid']]
                        T_w_r_mid = util.project_xy(session.get_T_w_r_odo(
                            t_mid).dot(T_r_mc)).dot(T_mc_r)
                        T_w_r_now = util.project_xy(session.get_T_w_r_odo(
                            t_now).dot(T_r_mc)).dot(T_mc_r)
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(T_r_m).dot(
                            polepos_m[imap][:, iactive])
                        desc = desclocal[imap][iactive]   # shape=(110,)
                        if quant:
                            desc = quantize_descriptor(desc, thresholds)
                        t0_ns = time.perf_counter_ns()
                        filter.update_measurement(desc, polepos_r_now[:2].T)
                        tim_update_measurement_ms[i] = (time.perf_counter_ns() - t0_ns) / 1e6
                        t0_ns = time.perf_counter_ns()
                        T_w_r_est[i] = filter.estimate_pose()
                        tim_estimate_pose_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6
                        if visualize:
                            polepos_w_est = T_w_r_est[i].dot(polepos_r_now)
                            locpoles.set_offsets(polepos_w_est[:2].T)
                    
                    ### knn filter measurement update ###   
                    if iactive:
                        t_mid = session.t_velo[locdata[imap]['imid']]
                        T_w_r_mid = util.project_xy(session.get_T_w_r_odo(
                            t_mid).dot(T_r_mc)).dot(T_mc_r)
                        T_w_r_now = util.project_xy(session.get_T_w_r_odo(
                            t_now).dot(T_r_mc)).dot(T_mc_r)
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(T_r_m).dot(
                            polepos_m[imap][:, iactive])
                        t0_ns = time.perf_counter_ns()
                        knn_filter.update_measurement_knn(polepos_r_now[:2].T)
                        tim_knn_measurement_ms[i] = (time.perf_counter_ns() - t0_ns) / 1e6

                        t0_ns = time.perf_counter_ns()
                        T_w_r_est_knn[i] = knn_filter.estimate_pose()
                        tim_knn_estimate_ms[i] += (time.perf_counter_ns() - t0_ns) / 1e6
                        if visualize:
                            polepos_w_est = T_w_r_est_knn[i].dot(polepos_r_now)
                            locpoles.set_offsets(polepos_w_est[:2].T)

                    imap += 1
            
            if visualize:
                particles.set_offsets(filter.particles[:, :2, 3])
                arrow.set_xy(T_w_r_est[i].dot(arrowdata)[:2].T)
                x, y = T_w_r_est[i, :2, 3]
                mapaxes.set_xlim(left=x - viewoffset, right=x + viewoffset)
                mapaxes.set_ylim(bottom=y - viewoffset, top=y + viewoffset)
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            tim_total_ms[i] = (time.perf_counter_ns() - t_step0_ns) / 1e6
            bar.update(i)
    filename = os.path.join(session.dir, get_locfileprefix() \
        + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.npz'))
    np.savez(filename, T_w_r_est=T_w_r_est, T_w_r_est_knn=T_w_r_est_knn, meas_events=np.array(meas_events, dtype=object),
            tim_total_ms=tim_total_ms, tim_update_measurement_ms=tim_update_measurement_ms, tim_estimate_pose_ms=tim_estimate_pose_ms,
            tim_knn_update_ms=tim_knn_update_ms, tim_knn_measurement_ms=tim_knn_measurement_ms, tim_knn_estimate_ms=tim_knn_estimate_ms,
            n_active_per_step=n_active_per_step,)
    
    print('append_count', append_count)

def plot_trajectories():
    trajectorydir = os.path.join(
        pynclt.resultdir, 'trajectories_est_{:.0f}_{:.0f}_{:.2f}'.format(
            n_mapdetections, n_locdetections, mapinterval))
    pgfdir = os.path.join(trajectorydir, 'pgf')
    util.makedirs(trajectorydir)
    util.makedirs(pgfdir)
    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans']
    for sessionname in pynclt.sessions:
        try:
            session = pynclt.session(sessionname)
            files = [file for file \
                in os.listdir(os.path.join(pynclt.resultdir, sessionname)) \
                    if file.startswith(get_locfileprefix())]
            for file in files:
                T_w_r_est = np.load(os.path.join(
                    pynclt.resultdir, sessionname, file))['T_w_r_est']
                plt.clf()
                plt.scatter(polemap[:, 0], polemap[:, 1], 
                    s=1, c='b', marker='.')
                plt.plot(session.T_w_r_gt[::20, 0, 3], 
                    session.T_w_r_gt[::20, 1, 3], color=(0.5, 0.5, 0.5))
                plt.plot(T_w_r_est[::20, 0, 3], T_w_r_est[::20, 1, 3], 'r')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.gcf().subplots_adjust(
                    bottom=0.13, top=0.98, left=0.145, right=0.98)
                filename = sessionname + file[18:-4]
                plt.savefig(os.path.join(trajectorydir, filename + '.svg'))
                plt.savefig(os.path.join(pgfdir, filename + '.pgf'))
        except:
            pass


def evaluate():
    stats = []
    for sessionname in pynclt.sessions:
        files = [file for file \
            in os.listdir(os.path.join(pynclt.resultdir, sessionname)) \
                if file.startswith(get_locfileprefix())]
        files.sort()

        session = pynclt.session(sessionname)

        cumdist = np.hstack([0.0, np.cumsum(np.linalg.norm(np.diff(
            session.T_w_r_gt[:, :3, 3], axis=0), axis=1))])
        t_eval = scipy.interpolate.interp1d(
            cumdist, session.t_gt)(np.arange(0.0, cumdist[-1], 1.0))
        T_w_r_gt = np.stack([util.project_xy(
                session.get_T_w_r_gt(t).dot(T_r_mc)).dot(T_mc_r) \
                    for t in t_eval])

        T_gt_est, T_gt_est_knn = [], []
        for file in files:
            fpath = os.path.join(pynclt.resultdir, sessionname, file)
            T_w_r_est = np.load(fpath)['T_w_r_est']  # [N_relodo, 4, 4]
            T_w_r_est_knn = np.load(fpath)['T_w_r_est_knn']  # knn filter
            T_w_r_est_interp = np.empty([len(t_eval), 4, 4])
            T_w_r_est_knn_interp = np.empty([len(t_eval), 4, 4]) # knn filter
            iodo = 1
            inum = 0
            for ieval in range(len(t_eval)):
                while session.t_relodo[iodo] < t_eval[ieval]:
                    iodo += 1
                    if iodo >= session.t_relodo.shape[0]:
                        break
                if iodo >= session.t_relodo.shape[0]:
                    break
                T_w_r_est_interp[ieval] = util.interpolate_ht(
                    T_w_r_est[iodo-1:iodo+1],
                    session.t_relodo[iodo-1:iodo+1], t_eval[ieval])
                
                # knn filter
                T_w_r_est_knn_interp[ieval] = util.interpolate_ht(
                    T_w_r_est_knn[iodo-1:iodo+1],
                    session.t_relodo[iodo-1:iodo+1], t_eval[ieval])
                inum += 1
            T_gt_est.append(
                np.matmul(util.invert_ht(T_w_r_gt), T_w_r_est_interp)[:inum, ...]
            )
            
            # knn filter
            T_gt_est_knn.append(
                np.matmul(util.invert_ht(T_w_r_gt), T_w_r_est_knn_interp)[:inum, ...]
            )
            
        T_gt_est = np.stack(T_gt_est) 
        T_gt_est_knn = np.stack(T_gt_est_knn) 
        L = T_gt_est.shape[1]
        t_plot = t_eval[:L]  

        poserrors = np.linalg.norm(T_gt_est[..., :2, 3], axis=-1)  # [n_runs, L]
        poserror_mean = np.mean(poserrors, axis=0)                 # [L]
        poserrors_knn = np.linalg.norm(T_gt_est_knn[..., :2, 3], axis=-1)  # [n_runs, L]
        poserror_mean_knn = np.mean(poserrors_knn, axis=0)                 # [L]

        # plot the evaluation result
        plot_evaluation_result(sessionname=sessionname, 
                               poserror_mean=poserror_mean, 
                               poserror_mean_knn=poserror_mean_knn,
                               t_plot=t_plot, 
                               files=files)

        lonerror = np.mean(np.mean(np.abs(T_gt_est[..., 0, 3]), axis=-1))
        laterror = np.mean(np.mean(np.abs(T_gt_est[..., 1, 3]), axis=-1))
        poserror = np.mean(np.mean(poserrors, axis=-1))
        poserror_knn = np.mean(np.mean(poserrors_knn, axis=-1)) # knn
        print(poserror, poserror_knn)
        posrmse = np.mean(np.sqrt(np.mean(poserrors**2, axis=-1)))
        angerrors = np.degrees(np.abs(
            np.array([util.ht2xyp(T)[:, 2] for T in T_gt_est])))
        angerror = np.mean(np.mean(angerrors, axis=-1))
        angrmse = np.mean(np.sqrt(np.mean(angerrors**2, axis=-1)))

        stats.append({'session': sessionname, 'lonerror': lonerror,
            'laterror': laterror, 'poserror': poserror, 'posrmse': posrmse,
            'angerror': angerror, 'angrmse': angrmse, 'T_gt_est': T_gt_est})

    np.savez(os.path.join(pynclt.resultdir, get_evalfile()), stats=stats)

    mapdata = np.load(os.path.join('nclt', get_globalmapname() + '.npz'))
    print('session \t f\te_pos \trmse_pos \te_ang \te_rmse')
    row = '{session} \t{f} \t{poserror} \t{posrmse} \t{angerror} \t{angrmse}'
    for i, stat in enumerate(stats):
        print(row.format(
            session=stat['session'],
            f=mapdata['mapfactors'][i] * 100.0,
            poserror=stat['poserror'],
            posrmse=stat['posrmse'],
            angerror=stat['angerror'],
            angrmse=stat['angrmse']))
        
def plot_evaluation_result(sessionname, poserror_mean, poserror_mean_knn, t_plot, files):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(f'Positional Error and Measurement Timestamps — {sessionname}')

    # ax1: baseline pos error
    ax1.plot(t_plot, poserror_mean, color='blue', label='poserror (mean over runs)')
    ax1.set_ylabel('poserror (m)')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right')

    # ax2: kNN pos error
    ax2.plot(t_plot, poserror_mean_knn, color='green', label='poserror (kNN mean over runs)')
    ax2.set_ylabel('poserror (m)')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')

    meas_times_all = []
    n_active_all = []
    for file in files:
        fpath = os.path.join(pynclt.resultdir, sessionname, file)
        data = np.load(fpath, allow_pickle=True)
        if 'meas_events' in data.files:
            events = data['meas_events']
            for e in events:
                try:
                    if isinstance(e, dict):
                        t_now = float(e.get('t_now'))
                        n_act = int(e.get('n_active'))
                    else:
                        ei = e.item() if hasattr(e, 'item') else {}
                        t_now = float(ei.get('t_now'))
                        n_act = int(ei.get('n_active'))
                    meas_times_all.append(t_now)
                    n_active_all.append(n_act)
                except Exception:
                    pass

    meas_times_all = np.array(meas_times_all, dtype=float) if len(meas_times_all) else np.empty((0,), dtype=float)
    n_active_all   = np.array(n_active_all, dtype=int)   if len(n_active_all)   else np.empty((0,), dtype=int)
    print(meas_times_all.shape, n_active_all.shape)

    if meas_times_all.size > 0:
        mask = (meas_times_all >= t_plot[0]) & (meas_times_all <= t_plot[-1])
        mt = meas_times_all[mask]
        na = n_active_all[mask]
    else:
        mt = np.empty((0,), dtype=float)
        na = np.empty((0,), dtype=int)

    # ax3
    ax3.set_xlabel('timestamp (t_eval)')
    ax3.set_ylabel('n_active')
    ax3.grid(True, linestyle='--', alpha=0.4)

    if na.size > 0:
        ax3.plot(mt, na, label='n_active (measurement)', linewidth=1.2)
        ax3.legend(loc='upper right')

        ymin = int(np.floor(na.min()))
        ymax = int(np.ceil(na.max()))
        ymin = min(ymin, 0)  
        if ymin == ymax:
            ymax = ymin + 1
        ax3.set_ylim(ymin - 0.5, ymax + 0.5)

        if (ymax - ymin) <= 20:
            ax3.set_yticks(range(ymin, ymax + 1))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    figpath = os.path.join(pynclt.resultdir, f"{sessionname}_poserror_meas.png")
    plt.savefig(figpath, dpi=300)
    plt.close(fig)
        
def merge_cluster_descriptors(poleparams, descs_array, threshold=0.2):
    """
    input:
      poleparams:  np.ndarray of shape (N, 3)
      descs_array: np.ndarray of shape (N, D)
      threshold:   float, 
    output:
      cluster_means: list of np.ndarray (shape (3,))  M x 3
      cluster_descs: list of np.ndarray (shape (D,))  M x D
    """
    # directly return input
    keep = np.any(descs_array != 0, axis=1)  
    return poleparams[keep], descs_array[keep]
    
    N, D = descs_array.shape

    # mask zero-only descriptors
    zero_mask = np.all(descs_array == 0, axis=1)
    zero_idxs = np.where(zero_mask)[0].tolist()
    nonzero_idxs = np.where(~zero_mask)[0].tolist()

    # sort by dim-0
    nonzero_idxs.sort(key=lambda i: descs_array[i, 0])

    clusters = []
    cluster_means = [] 
    cluster_mins  = []
    cluster_maxs  = []

    # cluster by order
    for idx in nonzero_idxs:
        desc_new = descs_array[idx]
        if not clusters:
            # the first cluster
            clusters.append([idx])
            cluster_means.append(desc_new.copy())
            cluster_mins.append(desc_new.copy())
            cluster_maxs.append(desc_new.copy())
        else:
            last = len(clusters) - 1
            mean = cluster_means[last]
            mins = cluster_mins[last]
            maxs = cluster_maxs[last]
            M = len(clusters[last])

            # try to combine
            can_merge = True
            for d in range(D):
                v0 = mean[d]
                v1 = desc_new[d]
                if v0 != 0 and v1 != 0:
                    new_mean = (v0 * M + v1) / (M + 1)
                    new_min  = min(mins[d],    v1)
                    new_max  = max(maxs[d],    v1)
                    if abs(new_mean - new_min) > threshold or \
                       abs(new_max - new_mean) > threshold:
                        can_merge = False
                        break

            if can_merge:
                # update the stat of new combined cluster
                clusters[last].append(idx)
                for d in range(D):
                    v0 = mean[d]
                    v1 = desc_new[d]
                    if v0 == 0 and v1 != 0:
                        mean[d] = v1
                    elif v0 != 0 and v1 != 0:
                        mean[d] = (v0 * M + v1) / (M + 1)
                        
                    if v1 != 0:
                        mins[d] = min(mins[d], v1)
                        maxs[d] = max(maxs[d], v1)
                
                cluster_means[last] = mean
                cluster_mins[last]  = mins
                cluster_maxs[last]  = maxs
            else:
                # create a new cluster
                clusters.append([idx])
                cluster_means.append(desc_new.copy())
                cluster_mins.append(desc_new.copy())
                cluster_maxs.append(desc_new.copy())

    # calculate the final cluster means and descs
    final_pole_means = []
    final_desc_means = []
    for k, idxs in enumerate(clusters):
        # pole means: include the pole for zero-only desc
        all_idxs = idxs + zero_idxs
        pm = np.mean(poleparams[all_idxs, :], axis=0)
        final_pole_means.append(pm)
        # desc means: cluster_means[k]
        final_desc_means.append(cluster_means[k].copy())

    # output np.ndarray
    if len(final_pole_means) == 0:
        return np.zeros((0, 3)), np.zeros((0, D))
    
    clustermeans  = np.vstack(final_pole_means)   # shape (M, 3)
    cluster_descs = np.vstack(final_desc_means)   # shape (M, D)
    return clustermeans, cluster_descs

def build_descmap_index(descmap, k):
    """
    divide descmap[:,0] to k parts, return:
      - descmap_index: dict[int, List[int]], bucket number -> global descriptor index list
      - edges: np.ndarray of shape (k+1,), edge number for dividing bucket

    Usage:
      descmap_index, edges = build_descmap_index(descmap, k=16)
    """
    dim0 = descmap[:, 0]
    edges = np.quantile(dim0, np.linspace(0, 1, k+1))
    
    descmap_index = {i: [] for i in range(k)}
    
    for j, v in enumerate(dim0):
        bin_idx = np.searchsorted(edges, v, side='right') - 1
        bin_idx = max(0, min(bin_idx, k-1))
        descmap_index[bin_idx].append(j)
    
    return descmap_index, edges

def quantize_descmap(descmap: np.ndarray, bits: int = 6):
    """
    Quantize a (N, 110) floating-point descriptor array into integer buckets
    using equal-frequency (quantile) binning.

    Parameters
    ----------
    descmap : np.ndarray, shape=(N, 110)
        Original floating-point descriptor array.
    bits : int
        Quantization bit-width. Total buckets = 2^bits.

    Returns
    -------
    qmap : np.ndarray, shape=(N, 110), dtype=int
    thresholds : np.ndarray, shape=(2^bits - 2,)
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")

    num_buckets = 1 << bits               # 2^bits
    if num_buckets < 2:
        # With 1 bucket total, only 0 would make sense; disallow
        raise ValueError("bits too small for quantization")

    flat = descmap.ravel()
    nonzero_mask = (flat != 0.0)
    nonzeros = flat[nonzero_mask]

    if nonzeros.size == 0:
        # All zeros → everything maps to bucket 0; thresholds are empty.
        return np.zeros_like(descmap, dtype=int), np.array([])

    # For non-zeros, we create (num_buckets - 1) buckets via (num_buckets - 2) thresholds.
    # Quantiles: 1/num_buckets, 2/num_buckets, ..., (num_buckets-1)/num_buckets
    quantiles = np.linspace(0.0, 1.0, num_buckets)[1:-1]  # exclude 0 and 1
    thresholds = np.quantile(nonzeros, quantiles)

    # Digitize non-zeros: indices in [0 .. len(thresholds)]
    idx = np.digitize(nonzeros, thresholds, right=False)

    # Shift by +1 to allocate codes 1..(num_buckets-1) for non-zeros
    qvals = idx + 1

    # Build output array
    qflat = np.zeros_like(flat, dtype=int)
    qflat[nonzero_mask] = qvals

    qmap = qflat.reshape(descmap.shape)
    return qmap, thresholds

def quantize_descriptor(vec: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Quantize a single descriptor vector (shape (110,)) using precomputed thresholds.

    Mapping:
      - Exact 0.0 → bucket 0
      - Non-zero values → np.digitize(value, thresholds) + 1
        which yields an integer bucket in [1, len(thresholds)+1]

    Parameters
    ----------
    vec : np.ndarray, shape=(110,)
        Floating-point descriptor vector.
    thresholds : np.ndarray, shape=(2^bits - 2,)
        Quantile thresholds returned by `quantize_descmap`.

    Returns
    -------
    q : np.ndarray, shape=(110,), dtype=int
        Quantized integer vector.
    """
    q = np.zeros_like(vec, dtype=int)
    nz = (vec != 0.0)
    if thresholds.size > 0:  # normal case
        idxs = np.digitize(vec[nz], thresholds, right=False)
        q[nz] = idxs + 1
    else:
        # thresholds empty means everything was zero in the dataset;
        # keep zeros (already initialized)
        pass
    return q

def plot_timing_stacked_for_sessions(
    sessions,
    base_dir="nclt",
    mask_measurement_only=True,
    max_bars=3000,
):
    """
    Stacked time series bar chart per session:
      x-axis: timestamp (from meas_events.t_now when available; else step index)
      y-axis: time per step (ms)
      stack:  update_measurement (bottom) + estimate_pose + kNN(total) + other

    Mask: by default only steps with tim_update_measurement_ms > 0.
    Downsampling: if steps > max_bars, group by stride and plot group means.

    Output per session: nclt/{session}/timing_stacked4_{session}.png
    """
    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localization*.npz")))
        if not paths:
            print(f"[timing] skip: no localization*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[timing] skip: failed to load {fpath}: {e}")
            continue

        required = ["tim_total_ms", "tim_update_measurement_ms", "tim_estimate_pose_ms"]
        if not all(k in data.files for k in required):
            print(f"[timing] skip: {fpath} missing required arrays {required}")
            continue

        total    = data["tim_total_ms"].astype(np.float32)
        update   = data["tim_update_measurement_ms"].astype(np.float32)
        estimate = data["tim_estimate_pose_ms"].astype(np.float32)

        # kNN parts are optional; default to zeros when missing
        knn_u = data["tim_knn_update_ms"].astype(np.float32)        if "tim_knn_update_ms" in data.files else np.zeros_like(total)
        knn_m = data["tim_knn_measurement_ms"].astype(np.float32)   if "tim_knn_measurement_ms" in data.files else np.zeros_like(total)
        knn_e = data["tim_knn_estimate_ms"].astype(np.float32)      if "tim_knn_estimate_ms" in data.files else np.zeros_like(total)
        knn = knn_u + knn_m + knn_e
        
        n_active = data["n_active_per_step"].astype(np.float32) if "n_active_per_step" in data.files else None

        # mask: only measurement steps by default
        mask = (update > 0.0) if mask_measurement_only else np.ones_like(update, dtype=bool)
        if not np.any(mask):
            print(f"[timing] skip: no steps after masking in {fpath}")
            continue

        total_m, update_m, estimate_m, knn_msk = total[mask], update[mask], estimate[mask], knn[mask]
        other_m = np.maximum(0.0, total_m - update_m - estimate_m - knn_msk)

        # timestamps: prefer meas_events.t_now if it matches, otherwise use masked indices
        ts = None
        if "meas_events" in data.files:
            try:
                ev = data["meas_events"]
                t_list = []
                for e in ev:
                    try:
                        d = e if isinstance(e, dict) else (e.item() if hasattr(e, "item") else {})
                        t_list.append(float(d.get("t_now")))
                    except Exception:
                        pass
                t_arr = np.array(t_list, dtype=float)
                if t_arr.size == total_m.size:
                    ts = t_arr
            except Exception:
                pass
        if ts is None:
            ts = np.flatnonzero(mask).astype(float)

        # downsample if too many bars (group means)
        n = total_m.size
        stride = int(np.ceil(n / max(1, max_bars)))
        if stride > 1:
            g = (n // stride) * stride

            def group_mean(a):
                return a[:g].reshape(-1, stride).mean(axis=1)

            ts_plot       = group_mean(ts)
            update_plot   = group_mean(update_m)
            estimate_plot = group_mean(estimate_m)
            knn_plot      = group_mean(knn_msk)
            other_plot    = group_mean(other_m)
            count_used    = ts_plot.size
            
            if n_active is not None:
                n_active_plot = group_mean(n_active[mask])
            else:
                n_active_plot = None
        else:
            ts_plot, update_plot, estimate_plot, knn_plot, other_plot = ts, update_m, estimate_m, knn_msk, other_m
            count_used = ts_plot.size
            n_active_plot = n_active[mask] if n_active is not None else None

        # bar width from timestamp spacing
        if ts_plot.size >= 2:
            dt = np.diff(ts_plot).mean()
            width = dt * 0.9 if dt > 0 else 1.0
        else:
            width = 1.0

        # stacked bars: update (bottom) + estimate + knn + other
        fig, ax = plt.subplots(figsize=(12, 5))
        b0 = update_plot
        b1 = b0 + estimate_plot
        b2 = b1 + knn_plot
        ax.bar(ts_plot, update_plot,   width=width, label="update_measurement (ms)")
        ax.bar(ts_plot, estimate_plot, width=width, bottom=b0, label="estimate_pose (ms)")
        ax.bar(ts_plot, knn_plot,      width=width, bottom=b1, label="kNN total (ms)")
        ax.bar(ts_plot, other_plot,    width=width, bottom=b2, label="other (ms)")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("time per step (ms)")
        ax.grid(True, linestyle="--", alpha=0.35)

        if n_active_plot is not None and n_active_plot.size == ts_plot.size:
            ax2 = ax.twinx()
            ax2.plot(ts_plot, n_active_plot, linewidth=1.0, color="k", alpha=0.7, label="n_active")
            ax2.set_ylabel("n_active (count)")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax.legend(loc="upper right")

        plt.tight_layout()
        out_path = os.path.join(sess_dir, f"timing_stacked4_{sess}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[timing] saved: {out_path}  (bars={count_used}, stride={stride})")
        
def report_max_timing_for_sessions(
    sessions,
    base_dir="nclt",
    mask_measurement_only=True,
    max_bars=3000, 
):
    """
    For each session:
      - load nclt/{session}/localization*.npz
      - mask to measurement steps if requested
      - find the step with maximum tim_total_ms
      - print timestamp and breakdown (update/estimate/kNN/other/total) to stdout
    """
    import os, glob
    import numpy as np

    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localization*.npz")))
        if not paths:
            print(f"[max-timing] skip: no localization*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[max-timing] skip: failed to load {fpath}: {e}")
            continue

        required = ["tim_total_ms", "tim_update_measurement_ms", "tim_estimate_pose_ms"]
        if not all(k in data.files for k in required):
            print(f"[max-timing] skip: {fpath} missing required arrays {required}")
            continue

        total    = data["tim_total_ms"].astype(np.float64)
        update   = data["tim_update_measurement_ms"].astype(np.float64)
        estimate = data["tim_estimate_pose_ms"].astype(np.float64)

        # kNN parts are optional
        knn_u = data["tim_knn_update_ms"].astype(np.float64)       if "tim_knn_update_ms" in data.files else np.zeros_like(total)
        knn_m = data["tim_knn_measurement_ms"].astype(np.float64)  if "tim_knn_measurement_ms" in data.files else np.zeros_like(total)
        knn_e = data["tim_knn_estimate_ms"].astype(np.float64)     if "tim_knn_estimate_ms" in data.files else np.zeros_like(total)
        knn = knn_u + knn_m + knn_e

        # mask: measurement-only or all steps
        mask = (update > 0.0) if mask_measurement_only else np.ones_like(update, dtype=bool)
        if not np.any(mask):
            print(f"[max-timing] skip: no steps after masking in {fpath}")
            continue

        total_m, update_m, estimate_m, knn_m = total[mask], update[mask], estimate[mask], knn[mask]
        other_m = np.maximum(0.0, total_m - update_m - estimate_m - knn_m)

        # timestamps: prefer meas_events.t_now if it matches masked length
        ts = None
        if "meas_events" in data.files:
            try:
                ev = data["meas_events"]
                t_list = []
                for e in ev:
                    try:
                        d = e if isinstance(e, dict) else (e.item() if hasattr(e, "item") else {})
                        t_list.append(float(d.get("t_now")))
                    except Exception:
                        pass
                t_arr = np.array(t_list, dtype=float)
                if t_arr.size == total_m.size:
                    ts = t_arr
            except Exception:
                pass
        if ts is None:
            ts = np.flatnonzero(mask).astype(float)  # fallback: masked step indices

        # find argmax on masked total
        k = int(np.argmax(total_m))
        tstamp   = float(ts[k])
        tot_ms   = float(total_m[k])
        upd_ms   = float(update_m[k])
        est_ms   = float(estimate_m[k])
        knn_ms   = float(knn_m[k])
        oth_ms   = max(0.0, tot_ms - upd_ms - est_ms - knn_ms)

        orig_idx = int(np.flatnonzero(mask)[k])

        print(
            f"[max-timing] session={sess}  orig_step={orig_idx}  timestamp={tstamp:.6f}  "
            f"total={tot_ms:.3f} ms  "
            f"update={upd_ms:.3f} ms  estimate={est_ms:.3f} ms  kNN={knn_ms:.3f} ms  other={oth_ms:.3f} ms"
        )
        
def plot_detect_timeline_for_sessions(
    sessions,
    base_dir="nclt",
    max_bars=3000,
    smooth_window=None,
):
    """
    Time-series bar chart of pole-detect time per local map, one figure per session.

    x-axis: timestamp per local map (from 'ts_localmaps' if present; else index)
    y-axis: detection time per local map (ms), drawn as bars
    Downsampling: if number of bars > max_bars, group by stride and plot group means
    Optional smoothing line: running mean with a centered window (smooth_window)

    Input NPZ (first match of 'localmaps*.npz' in the session dir) is expected to contain:
      - tim_detect_ms: float array of shape (N_localmaps,)
      - ts_localmaps:  float array of shape (N_localmaps,)   [optional but recommended]
    """
    def running_mean(a, w):
        if w is None or w <= 1:
            return a
        # centered running mean; trim edges to keep same length as a
        k = w // 2
        c = np.cumsum(np.pad(a, (1, 0), mode="constant"))
        m = (c[w:] - c[:-w]) / float(w)
        # pad to original length
        left  = np.full(k, m[0], dtype=a.dtype)
        right = np.full(a.size - m.size - k, m[-1], dtype=a.dtype)
        return np.concatenate([left, m, right])

    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localmaps*.npz")))
        if not paths:
            print(f"[detect-timeline] skip: no localmaps*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[detect-timeline] skip: failed to load {fpath}: {e}")
            continue

        if "tim_detect_ms" not in data.files:
            print(f"[detect-timeline] skip: {fpath} missing 'tim_detect_ms'")
            continue

        detect = data["tim_detect_ms"].astype(np.float32)
        # Preferred timestamps if present
        if "ts_localmaps" in data.files:
            ts = data["ts_localmaps"].astype(np.float64)
        else:
            # Fallback: use dense indices as x
            ts = np.arange(detect.size, dtype=np.float64)

        if ts.size != detect.size:
            # Size mismatch safety: fall back to indices
            print(f"[detect-timeline] warn: ts length != detect length in {fpath}; fallback to index x-axis")
            ts = np.arange(detect.size, dtype=np.float64)

        n = detect.size
        stride = int(np.ceil(n / max(1, max_bars)))
        if stride > 1:
            # group means to reduce bars
            g = (n // stride) * stride

            def group_mean(a):
                return a[:g].reshape(-1, stride).mean(axis=1)

            ts_plot     = group_mean(ts)
            detect_plot = group_mean(detect)
            count_used  = ts_plot.size
        else:
            ts_plot, detect_plot = ts, detect
            count_used = ts_plot.size

        # Choose a reasonable bar width from timestamp spacing
        if ts_plot.size >= 2:
            dt = np.diff(ts_plot).mean()
            width = dt * 0.9 if dt > 0 else 1.0
        else:
            width = 1.0

        # Optional smoothing line (running mean)
        smooth = running_mean(detect_plot, smooth_window)

        # Plot
        plt.figure(figsize=(12, 4.5))
        plt.bar(ts_plot, detect_plot, width=width, label="detect time (ms)")
        if smooth_window and smooth_window > 1:
            plt.plot(ts_plot, smooth, linewidth=1.25, label=f"running mean (w={smooth_window})")
        title = (f"Pole-detect time over time — {sess}\n"
                 f"N={int(n)} local maps{' (downsampled)' if stride>1 else ''}")
        plt.title(title)
        plt.xlabel("timestamp")
        plt.ylabel("time per local map (ms)")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(loc="upper right")
        plt.tight_layout()

        out_path = os.path.join(sess_dir, f"detect_timeline_{sess}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[detect-timeline] saved: {out_path}  (bars={count_used}, stride={stride})")
        
        
def report_pole_detect_timing_for_sessions(
    sessions,
    base_dir="nclt",
    drop_zeros=True,
    percentiles=(90, 95, 99),
):
    """
    For each session, read nclt/{session}/localmaps_*.npz and print a timing summary
    for tim_detect_ms:
      - N / mean / median / std / min / {percentiles} / max
      - Index of the max in the original array (local map index)
    """
    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localmaps_*.npz")))
        if not paths:
            print(f"[detect-report] skip: no localmaps*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[detect-report] skip: failed to load {fpath}: {e}")
            continue

        if "tim_detect_ms" not in data.files:
            print(f"[detect-report] skip: {fpath} has no 'tim_detect_ms'")
            continue

        arr = data["tim_detect_ms"].astype(np.float32)

        # Keep finite values; optionally drop zeros.
        mask = np.isfinite(arr)
        if drop_zeros:
            mask &= (arr > 0.0)

        if not np.any(mask):
            print(f"[detect-report] skip: no valid times after masking in {fpath}")
            continue

        t = arr[mask]
        n = t.size
        mean_v = float(np.mean(t))
        med_v  = float(np.median(t))
        std_v  = float(np.std(t))
        min_v  = float(np.min(t))
        max_v  = float(np.max(t))

        # Map the argmax in the masked array back to the original index.
        idx_in_masked = int(np.argmax(t))
        orig_indices  = np.flatnonzero(mask)
        max_idx_orig  = int(orig_indices[idx_in_masked])

        # Requested percentiles.
        pct_vals = {}
        for p in percentiles:
            try:
                pct_vals[p] = float(np.percentile(t, p))
            except Exception:
                pct_vals[p] = float("nan")

        pcts_str = " ".join([f"p{p}={pct_vals[p]:.2f}ms" for p in percentiles])
        print(
            f"[detect-report] session={sess}  N={n}  mean={mean_v:.2f}ms  median={med_v:.2f}ms  "
            f"std={std_v:.2f}ms  min={min_v:.2f}ms  max={max_v:.2f}ms (map_idx={max_idx_orig})  {pcts_str}"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant', nargs='?', const=6, type=int, help='Enable quant mode with specified bits (default: 6)')
    parser.add_argument('--nlist', type=int, default=None,
                        help='FAISS IVF nlist (number of centroids/lists). '
                             'Default: use KMeansIVF default.')
    parser.add_argument('--nprobe', type=int, default=None,
                        help='FAISS IVF nprobe (lists to probe per query). '
                             'Default: use KMeansIVF default.')
    args = parser.parse_args()
    
    if args.quant is None:
        do_quant = False
        quant_bits = None
    else:
        do_quant = True
        quant_bits = args.quant
        
    print(f"Quantization enabled? {do_quant}, bits={quant_bits}")
    ivf_nlist  = args.nlist
    ivf_nprobe = args.nprobe
    
    #save_global_map(use_desc=True)
    for session in pynclt.sessions:
        #save_local_maps(session, use_desc=True)
        localize(session, visualize=False, quant=do_quant, quant_bits=quant_bits, ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe)

    #plot_trajectories()
    evaluate()
    
    plot_timing_stacked_for_sessions(pynclt.sessions, base_dir="nclt")
    report_max_timing_for_sessions(pynclt.sessions, base_dir="nclt")
    # plot_detect_timeline_for_sessions(pynclt.sessions, base_dir="nclt")
    # report_pole_detect_timing_for_sessions(pynclt.sessions, base_dir="nclt")
    
    
# for session in sessions:
#     for all mappos do clustermeans, clusterdescs calculation
#     if this is the first session:
#         add all of the clusterdescs and clustermeans into the global map
#     else:
#         for each desc in cluster descs:
#             do_match(desc, globaldescs)
#             if the matched pair <i, j> satisfy distance(clustermeans[i], globalpolexy[j]) < 1.0:
#                 continue
#             else:
#                 add the desc and corresponding pole clustermean into the global map
