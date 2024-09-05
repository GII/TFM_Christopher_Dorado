# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2023 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
# Acquisition.py shows how to acquire images. It relies on
# information provided in the Enumeration example. Also, check out
# the ExceptionHandling and NodeMapInfo examples if you haven't already.
# ExceptionHandling shows the handling of standard and Spinnaker exceptions
# while NodeMapInfo explores retrieving information from various node types.
#
# This example touches on the preparation and cleanup of a camera just before
# and just after the acquisition of images. Image retrieval and conversion,
# grabbing image data, and saving images are all covered as well.
#
# Once comfortable with Acquisition, we suggest checking out
# AcquisitionMultipleCamera, NodeMapCallback, or SaveToAvi.
# AcquisitionMultipleCamera demonstrates simultaneously acquiring images from
# a number of cameras, NodeMapCallback serves as a good introduction to
# programming with callbacks and events, and SaveToAvi exhibits video creation.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import os
import PySpin
import sys
import Constantes

class StreamMode:
    STREAM_MODE_TELEDYNE_GIGE_VISION = 0
    STREAM_MODE_PGRLWF = 1
    STREAM_MODE_SOCKET = 2

CHOSEN_STREAMMODE = StreamMode.STREAM_MODE_TELEDYNE_GIGE_VISION
NUM_IMAGES = 1

def set_stream_mode(cam):
    streamMode = "TeledyneGigeVision" if CHOSEN_STREAMMODE == StreamMode.STREAM_MODE_TELEDYNE_GIGE_VISION else "LWF" if CHOSEN_STREAMMODE == StreamMode.STREAM_MODE_PGRLWF else "Socket"
    result = True

    nodemap_tlstream = cam.GetTLStreamNodeMap()
    node_stream_mode = PySpin.CEnumerationPtr(nodemap_tlstream.GetNode('StreamMode'))

    if not PySpin.IsReadable(node_stream_mode) or not PySpin.IsWritable(node_stream_mode):
        return True

    node_stream_mode_custom = PySpin.CEnumEntryPtr(node_stream_mode.GetEntryByName(streamMode))

    if not PySpin.IsReadable(node_stream_mode_custom):
        print('Stream mode ' + streamMode + ' not available. Aborting...')
        return False

    stream_mode_custom = node_stream_mode_custom.GetValue()
    node_stream_mode.SetIntValue(stream_mode_custom)

    print('Stream Mode set to %s...' % node_stream_mode.GetCurrentEntry().GetSymbolic())
    return result

def print_pixel_formats(nodemap):
    print('*** PIXEL FORMATS ***\n')

    try:
        node_pixel_formats = PySpin.CCategoryPtr(nodemap.GetNode('PixelFormat'))

        if PySpin.IsReadable(node_pixel_formats):
            entries = node_pixel_formats.GetEntries()
            for entry in entries:
                node_entry = PySpin.CEnumEntryPtr(entry)
                if PySpin.IsReadable(node_entry):
                    print('Pixel Format: %s' % node_entry.GetSymbolic())

        else:
            print('Pixel format information not readable.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)

def set_pixel_format(cam, format_name):
    try:
        nodemap = cam.GetNodeMap()
        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))

        if not PySpin.IsReadable(node_pixel_format) or not PySpin.IsWritable(node_pixel_format):
            print('Unable to access pixel format node. Aborting...')
            return False

        node_pixel_format_entry = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(format_name))

        if not PySpin.IsReadable(node_pixel_format_entry):
            print('Pixel format %s not available. Aborting...' % format_name)
            return False

        pixel_format_value = node_pixel_format_entry.GetValue()
        node_pixel_format.SetIntValue(pixel_format_value)

        print('Pixel Format set to %s...' % format_name)
        return True

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

def set_exposure(cam, exposure_time_us):
    try:
        nodemap = cam.GetNodeMap()
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))

        if PySpin.IsReadable(node_exposure_time) and PySpin.IsWritable(node_exposure_time):
            node_exposure_time.SetValue(exposure_time_us)
            print(f'Exposure time set to {exposure_time_us} us...')
            return True
        else:
            print('Unable to set exposure time.')
            return False

    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False

def set_brightness(cam, brightness_value):
    try:
        nodemap = cam.GetNodeMap()
        node_brightness = PySpin.CIntegerPtr(nodemap.GetNode('Brightness'))

        if PySpin.IsReadable(node_brightness) and PySpin.IsWritable(node_brightness):
            node_brightness.SetValue(brightness_value)
            print(f'Brightness set to {brightness_value}...')
            return True
        else:
            print('Unable to set brightness.')
            return False

    except PySpin.SpinnakerException as ex:
        print(f'Error: {ex}')
        return False

def acquire_images(cam, nodemap, nodemap_tldevice):
    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False

        node_acquisition_mode_continuous = PySpin.CEnumEntryPtr(node_acquisition_mode.GetEntryByName('Continuous'))
        if not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print('Acquisition Mode set to Continuous...')

        cam.BeginAcquisition()

        device_serial_number = ''
        node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
        if PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        for i in range(NUM_IMAGES):
            try:
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    print('Grabbed image with width = %d, height = %d' % (width, height))

                    image_converted = processor.Convert(image_result, PySpin.PixelFormat_RGB8)

                    # Define the path and filename
                    if device_serial_number:
                        filename = os.path.join(Constantes.image_path, 'Acquisition-%s-%d.jpg' % (device_serial_number, i))
                    else:
                        filename = os.path.join(Constantes.image_path, 'Acquisition-%d.jpg' % i)

                    # Ensure the save directory exists
                    if not os.path.exists(Constantes.image_path):
                        os.makedirs(Constantes.image_path)

                    image_converted.Save(filename)
                    print('Image saved at %s' % filename)

                    image_result.Release()
                    print('')

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def print_device_info(nodemap):
    print('*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def run_single_camera(cam):
    try:
        result = True
        print("Inicializando cámara...")
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        result &= print_device_info(nodemap_tldevice)

        print("Configurando cámara...")
        cam.Init()

        nodemap = cam.GetNodeMap()

        result &= set_stream_mode(cam)

        print_pixel_formats(nodemap)

        result &= set_pixel_format(cam, 'RGB8')

        result &= set_exposure(cam, 20000)
        result &= set_brightness(cam, 128)  # Adjust brightness if needed

        print("Adquiriendo imágenes...")
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        print("Desinicializando cámara...")
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error en la cámara: %s' % ex)
        result = False

    return 0

def main():
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('No se puede escribir en el directorio actual. Verifica permisos.')
        input('Presiona Enter para salir...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True

    try:
        system = PySpin.System.GetInstance()
        version = system.GetLibraryVersion()
        print('Versión de la biblioteca: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        cam_list = system.GetCameras()
        num_cameras = cam_list.GetSize()

        print('Número de cámaras detectadas: %d' % num_cameras)

        if num_cameras == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            print('¡No se encontraron cámaras!')
            input('¡Listo! Presiona Enter para salir...')
            return False

        for i, cam in enumerate(cam_list):
            print('Ejecutando ejemplo para la cámara %d...' % i)
            result &= run_single_camera(cam)
            print('Ejemplo de la cámara %d completo... \n' % i)

        del cam
        cam_list.Clear()
        system.ReleaseInstance()

    except PySpin.SpinnakerException as ex:
        print('Error en el sistema: %s' % ex)
        result = False

    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
