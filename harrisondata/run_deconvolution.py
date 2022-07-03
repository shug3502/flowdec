import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import bioformats.omexml as ome
import tifffile
import sys
import argparse
import os
import logging
from skimage import exposure, external
from scipy import ndimage, signal, stats
from math import floor, ceil
from flowdec import data as fd_data
from flowdec import psf as fd_psf
from flowdec import restoration as fd_restoration
from xml.etree import cElementTree as ElementTree

#deal with command line inputs
parser = argparse.ArgumentParser(description='Deconvolve LLSM data.') 
parser.add_argument('input_file', metavar='f', type=str, nargs=1, 
                   help='path to an .ome.tif image stack to deconvolve')
parser.add_argument('--bead', type=str, nargs=1, default = None,
                   help='path to a .csv file specifying Sigma as covariance matrix to use for PSF')
parser.add_argument('--niter', dest='niter', type=int, default=30,
                   help='number of iterations to run the Richardson-Lucy deconvolution algorithm for')
parser.add_argument('--shape', dest='kernel_shape', type=int, nargs='?', default=[51,51,51],
                   help='shape of the kernel for PSF')
args = parser.parse_args()
input_file_str = args.input_file[0]
output_file_str = os.path.splitext(os.path.splitext(input_file_str)[0])[0] + '_flowdec_deconvolved.ome.tif'
bead_image_file_str = args.bead
kernel_shape = args.kernel_shape
niter = args.niter

if bead_image_file_str is None:
    bead_image_file_str = input_file_str[0:35] + "PSF_bead_image_sigma.csv" #/harrisondata/OS_LLSM_191206_MC191_ with appropriate date in middle
else: 
    bead_image_file_str = bead_image_file_str[0]

if os.path.isfile(input_file_str) & os.path.isfile(bead_image_file_str):
    msg = 'Preparing to deconvole the image stack ' + input_file_str + ' using a PSF derived from the bead image at ' + bead_image_file_str
    print(msg)
else:
    raise Exception('Unable to find files at paths for input image or bead')

#set logging level: how much info to print out
#see https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#reference for metadata stuff:
#https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary
class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text) 

class XmlDictConfig(dict):
    '''
    Example usage:
    >>> tree = ElementTree.parse('your_file.xml') root = tree.getroot() 
    >>> xmldict = XmlDictConfig(root)
    Or, if you want to use an XML string:
    >>> root = ElementTree.XML(xml_string) xmldict = XmlDictConfig(root)
    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags 
                # in a series are different, then they are all 
                # different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags 
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is 
                    # the tag name the list elements all share in 
                    # common, and the value is the list itself
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag, you 
            # won't be having any text. This may or may not be a good 
            # idea -- time will tell. It works for the way we are 
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, 
            # extract the text
            else:
                self.update({element.tag: element.text})

def writeplanes(pixel, SizeT=1, SizeZ=1, SizeC=1, order='TZCYX', verbose=False):
    if order == 'TZCYX':
        p.DimensionOrder = ome.DO_XYCZT
        counter = 0
        for t in range(SizeT):
            for z in range(SizeZ):
                for c in range(SizeC):
                    if verbose:
                        print('Write PlaneTable: ', t, z, c),
                        sys.stdout.flush()
                    pixel.Plane(counter).TheT = t
                    pixel.Plane(counter).TheZ = z
                    pixel.Plane(counter).TheC = c
                    counter = counter + 1
    return pixel

#read metadata and write to ome.tif file
with tifffile.TiffFile(input_file_str) as tif:
    imagej_hyperstack = tif.asarray()
    imagej_metadata = tif.imagej_metadata
    xml_string = tif.ome_metadata 
root = ElementTree.XML(xml_string) 
xml_dict = XmlDictConfig(root)
PhysicalSizeX = float(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['PhysicalSizeX']) 
PhysicalSizeY = float(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['PhysicalSizeY']) 
PhysicalSizeZ = float(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['PhysicalSizeZ']) 
TimeIncrement = float(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['TimeIncrement']) 
SizeC = int(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['SizeC']) 
SizeT = int(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['SizeT']) 
SizeZ = int(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['SizeZ']) 
SizeY = int(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['SizeY']) 
SizeX = int(xml_dict['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Image']['{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels']['SizeX']) 
is_single_time_pt = (SizeT>1)
# Dimension TZCXY Sizes set from reading in metadata
Series = 0
pixeltype = 'uint8'
dimorder = 'TZCYX'
# Getting metadata info
omexml = ome.OMEXML()
omexml.image(Series).Name = output_file_str
p = omexml.image(Series).Pixels
#p.ID = 0
p.SizeX = SizeX 
p.SizeY = SizeY 
p.SizeC = SizeC 
p.SizeT = SizeT 
p.SizeZ = SizeZ 
p.TimeIncrement = np.float(TimeIncrement) 
p.PhysicalSizeX = np.float(PhysicalSizeX) 
p.PhysicalSizeY = np.float(PhysicalSizeY) 
p.PhysicalSizeZ = np.float(PhysicalSizeZ)
p.PixelType = pixeltype
p.channel_count = SizeC 
p.plane_count = SizeZ * SizeT * SizeC 
p = writeplanes(p, SizeT=SizeT, SizeZ=SizeZ, SizeC=SizeC, order=dimorder) 
for c in range(SizeC):
    if pixeltype == 'unit8':
        p.Channel(c).SamplesPerPixel = 1
    if pixeltype == 'unit16':
        p.Channel(c).SamplesPerPixel = 2 
omexml.structured_annotations.add_original_metadata(ome.OM_SAMPLES_PER_PIXEL, str(SizeC))
# Converting to omexml
xml = omexml.to_xml()
print("Metadata successfully read \n")
#################################
#now actually read the data
dat = external.tifffile.imread(input_file_str)
psf = np.genfromtxt(bead_image_file_str, delimiter=',')
print("Data successfully read in \n")

kernel = np.zeros(kernel_shape) #(51,51,51)) #Note may not work if this size is too big relative to the image
for offset in [0,1]:
    kernel[tuple((np.array(kernel.shape) - offset) // 2)] = 1
#assume bead image at different resolution in z to actual data, and difference is a factor of 3.08 (308nm versus 100nm)
#estimated and stored 3d psf, but assume here diagonal psf. Could adjust this, but already very close to diagonal
kernel = ndimage.gaussian_filter(kernel, sigma=[np.sqrt(psf[2,2])/3.08,np.sqrt(psf[0,0]),np.sqrt(psf[1,1])])

############

data = np.copy(dat) 
res = np.zeros(data.shape) 
ndim = 3 #data.ndim 
algo = fd_restoration.RichardsonLucyDeconvolver(ndim, pad_mode='none', pad_min=[16,16,16]).initialize() 
tmp = algo.run(fd_data.Acquisition(data=data[0], kernel=kernel), niter=2)
for tt in range(SizeT):
    if tt%10==0:
        print(str(tt)+' of ' + str(SizeT) + ' complete')
    nonzero = data[tt][np.where(data[tt]>0)]
    bkgd_mode = stats.mode(nonzero)[0][0]
    bkgd_std = stats.tstd(nonzero)
    indz,indy,indx = np.where(data[tt]==0)
    data[tt,indz, indy, indx] = bkgd_mode + bkgd_std*np.random.randn(len(indz))
    # Note that deconvolution initialization is best kept separate from 
    # execution since the "initialize" operation corresponds to creating 
    # a TensorFlow graph, which is a relatively expensive operation and 
    # should not be repeated across iterations if deconvolving more than 
    # one image
    res[tt] = algo.run(fd_data.Acquisition(data=data[tt], kernel=kernel), niter=niter).data

####################
print('Done. Now writing to file\n')
# create numpy array with correct order
if is_single_time_pt:
    img5d = np.expand_dims(np.expand_dims(res, axis=1), axis=0) 
else:
    img5d = np.expand_dims(res, axis=1)
# write file and save OME-XML as description
tifffile.imwrite(output_file_str, img5d, metadata={'axes': dimorder}, description=xml)
print('All finished\n')



