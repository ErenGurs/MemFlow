import numpy
import sys
import argparse
import os
import torch
sys.path.append('core')
from utils import flow_viz
import PIL
from PIL import Image


# Utility class to deal with the Optical Flow
class FlowUtils:

    def read_flo(self, str_file):
        str_flow = open(str_file, 'rb').read()

        assert(numpy.frombuffer(str_flow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

        width = numpy.frombuffer(str_flow, dtype=numpy.int32, count=1, offset=4)[0]
        height = numpy.frombuffer(str_flow, dtype=numpy.int32, count=1, offset=8)[0]

        return numpy.frombuffer(
            str_flow, dtype=numpy.float32, count=height * width * 2, offset=12).reshape([height, width, 2])
    # end

    # Save flow in numpy array as colorwheel
    def save_color_flow(self, flow, name):
        flow_color = flow_viz.flow_to_color(flow, convert_to_bgr=False)
        PIL.Image.fromarray(flow_color).save(name + '.png')

    # Save *.flo file as colorwheel (*.png)
    def save_color_flow(self, flo_file, out_cw_file=None):
        if out_cw_file == None:
            # Save to current directory using the .flo filename but with .png extension
            out_cw_file = os.path.splitext(os.path.basename(flo_file))[0] + '.png'

        numpy_forward = self.read_flo(flo_file)
        flow_img = flow_viz.flow_to_image(numpy_forward)  # if tensor convert to numpy: .permute(1, 2, 0).numpy()
        image = Image.fromarray(flow_img)
        print ("Saving colorwheel image " + out_cw_file )
        image.save(out_cw_file)


#
# Create folder structure from BMS data:
#  For ex: The following files (pictures and flow)
#      val2_non-rigid_001-flow_01.flo
#      val2_non-rigid_001-img_0.png
#      val2_non-rigid_001-img_1.png
#    will be copied to the following newly created folder. It is easier to find
#    optical flow, confidence maps etc using external tools such as MemFlow. Besides
#    it is better to have a directory per picture pair (and its forward flow) 
#      val2_non-rigid_001/
#
def process_bms_data(bms_data_location, out_data_location=None):
    if out_data_location == None:
        out_data_location = "./bms_data/"

    # Use FlowUtils package
    fu = FlowUtils()

    dirList = os.listdir(bms_data_location)
    for fname in dirList:

        # Create folders from fname (file name)
        ext = os.path.splitext(fname)
        labels = ext[0].rsplit('-', 1)  # split max=1 from the last delimiter "-"
        out_folder = out_data_location + "/" + labels[0]
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print("Created directory " + out_folder)

        # input and destination filenames
        fname_with_path = bms_data_location + '/' + fname 
        dest_fname_with_path = out_folder + '/' + fname
        if not os.path.exists(dest_fname_with_path): 
            # Copy BMS files to under the new directory structure
            copy_command = 'cp ' + fname_with_path + ' ' + dest_fname_with_path
            os.system (copy_command)

        # Save colorwheel (.png) from the *.flo file
        if ext[1] == '.flo':
            png_with_path = out_folder + "/" + ext[0] + '.png'
            if not os.path.exists(png_with_path):
                fu.save_color_flow(fname_with_path, png_with_path)

            #numpy_forward = fu.read_flo(fname_with_path)
            #tensor_forward = torch.from_numpy(numpy_forward).float().cuda().permute(2,0,1).unsqueeze(dim=0)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate colorwheel of the input *.flo")
    parser.add_argument('filename', help='*.flo filename')
    #print ('argument list', sys.argv)
    #filename = sys.argv[1]
    args = parser.parse_args()
    filename = os.path.basename(args.filename)
    location = os.path.dirname(args.filename)
    
    # Process BMS data
    process_bms_data(location)



    # Save colorwheel file
    #out_filename = os.path.splitext(filename)[0] + '.png'
    #numpy_forward = read_flo(args.filename)
    #flow_img = flow_viz.flow_to_image(numpy_forward)  # if tensor convert to numpy: .permute(1, 2, 0).numpy()
    #image = Image.fromarray(flow_img)
    #print ("Saving colorwheel image " + out_filename )
    #image.save(out_filename)