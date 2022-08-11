from leb.imajin.models.detectors.cmos_camera import CMOSCamera

def test_CMOSCamera():
    num_pixels = (128, 128) 
    
    camera = CMOSCamera(num_pixels)

    assert camera.num_pixels == (128, 128)
