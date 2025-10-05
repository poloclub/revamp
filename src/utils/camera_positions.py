
import mitsuba as mi
import numpy as np
import os
import csv
import warnings

def generate_stop_sign_approach_cam_moves(sample_size=10) -> np.array:
    """
    Generate an np.ndarray of camera transform matrices
    Read in a set of camera positions that comprise an animation of the camera position
    """
    # grab animation values of the camera generated in blender aand 
    # construct transform matrices for each camera position we want to sample
    raise DeprecationWarning("This function is deprecated. Use the scene configuration file to specify sensor parameters.")

    scene_path = os.path.join("scenes", "intersection_taxi")
    animations_path = os.path.join(scene_path,"animations", "cam_moves.csv")
    data = csv.reader(open(animations_path))
    # invert values for blender/mitsuba compatability
    moves = np.array([-float(d[0]) for d in data][0:]) 
    sample_moves = np.random.choice(moves, sample_size) #randomly sample cameraa positions (default=10)
    moves_matrices = []
    mat = p[k1].matrix
    for m in sample_moves:
        _mat = mi.cuda_ad_rgb.Matrix4f(mat)
        _mat[3][2]= m # modify the camera z-position
        moves_matrices.append(_mat)
    return np.array(moves_matrices)


def generate_taxi_cam_positions() -> np.array:
    def load_sensor(r, y, phi, theta):
        from mitsuba import ScalarTransform4f as T
        # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
        origin = T.rotate([0, 1, 0], phi).rotate([0, 0, 1], theta) @ mi.ScalarPoint3f([0, y, r])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, -0.20, 0],
                #up=[0, 0, 1]
                up=[0, 1, 0]            
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })

    raise DeprecationWarning("This function is deprecated. Use the scene configuration file to specify sensor parameters.")

    # e.g, use sensor_count=6 & phis=30 to get 180 deg view
    sensor_count = 6
    radius = 5.0
    phis = [60.0 * i for i in range(sensor_count)]
    theta = 12.0
    # y = 5
    # ys = [1,3,5]
    ys = [1]
    sensors = np.array([])
    for y in ys:
        _sensors = np.array([load_sensor(radius, y, phi, theta) for phi in phis])
        sensors = np.append(sensors, _sensors)
    sensors = sensors.flatten()
    sensors = np.array([s.world_transform() for s in sensors])
    return sensors

def generate_sunset_taxi_cam_positions() -> np.array:
    mi.load_file("scenes/street_sunset/street_sunset.xml")
    p = mi.traverse(scene)
    cam_keys = ['PerspectiveCamera_5.to_world', \
        'PerspectiveCamera.to_world', 
        'PerspectiveCamera_1.to_world', 
        'PerspectiveCamera_2.to_world',
        'PerspectiveCamera_3.to_world',
        'PerspectiveCamera_4.to_world',         
        'PerspectiveCamera_6.to_world',
        'PerspectiveCamera_7.to_world']
    sensors = np.array([p[k] for k in cam_keys])
    return sensors


def generate_cube_scene_cam_positions() -> np.array:
    """
    Load a mesh and use its vertices as camera positions
    e.g.,  Load a half-icosphere and separate the vertices by their height above target object
    each strata of vertices forms a 'ring' around the object. place cameras in a ring around the object
    and return camera positions (world_transform())
    """
    raise DeprecationWarning("This function is deprecated. Use the scene configuration file to specify sensor parameters.")

    from mitsuba import ScalarTransform4f as T    
    def load_sensor_at_position(x,y,z):  
        origin = mi.ScalarPoint3f([x,y,z])

        return mi.load_dict({
            'type': 'perspective',
            'fov': 39.3077,
            'to_world': T.look_at(
                origin=origin,
                target=[0, -0.5, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': 16
            },
            'film': {
                'type': 'hdrfilm',
                'width': 512,
                'height': 512,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })
    sphere = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_mid.ply"
        },
    })
    sphere_outer = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_outer.ply"
        },
    })  
    sphere_inner = mi.load_dict({
        'type': 'scene',
        'sphere': {
            'type': 'ply',
            'filename': "scenes/cube_scene/meshes/sphere_inner.ply"
        },
    })        
    ip = mi.traverse(sphere)
    ipv = np.array(ip["sphere.vertex_positions"])
    ipv  = np.reshape(ipv,(int(len(ipv)/3),3))    

    outer_sphere_ip = mi.traverse(sphere_outer)
    outer_sphere_ipv = np.array(outer_sphere_ip["sphere.vertex_positions"])
    outer_sphere_ipv = np.reshape(outer_sphere_ipv,(int(len(outer_sphere_ipv)/3),3))    

    inner_sphere_ip = mi.traverse(sphere_inner)
    inner_sphere_ipv = np.array(inner_sphere_ip["sphere.vertex_positions"])    
    inner_sphere_ipv = np.reshape(inner_sphere_ipv,(int(len(inner_sphere_ipv)/3),3))       
    # strata = np.array(list(set(np.round(ipv[:,1],3))))  
    # strata_2_cams =  ipv[np.where(np.round(ipv,3)[:,1] == strata[2])]    
    # strata_1_cams = ipv[np.where(np.round(ipv,3)[:,1] == strata[1])]    
    ipv_f = ipv[np.where(ipv[:,0] > 0)]
    outer_sphere_ipv_f = outer_sphere_ipv[np.where(outer_sphere_ipv[:,0] > 0)]
    inner_sphere_ipv_f = inner_sphere_ipv[np.where(inner_sphere_ipv[:,0] > 0)]
    cam_pos_ring = np.concatenate((ipv_f, outer_sphere_ipv_f, inner_sphere_ipv_f))
    positions = np.array([load_sensor_at_position(p[0], p[1], p[2]).world_transform() for p in cam_pos_ring])
    return positions

