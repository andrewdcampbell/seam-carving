from seam_carving import SeamCarver
import time
import os



def image_resize_without_mask(filename_input, filename_output, dy, dx):
    obj = SeamCarver(filename_input, dy, dx)
    obj.save_result(filename_output)


def image_resize_with_mask(filename_input, filename_output, dy, dx, filename_mask):
    obj = SeamCarver(filename_input, dy, dx, protect_mask=filename_mask)
    obj.save_result(filename_output)


def object_removal(filename_input, filename_output, filename_mask):
    obj = SeamCarver(filename_input, 0, 0, object_mask=filename_mask)
    obj.save_result(filename_output)



if __name__ == '__main__':
    """
    Put image in in/images folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """
    input_image = "demos/train.jpg"
    output_image = "testy.jpg"
    input_mask = "demos/train_mask.jpg"
    dy, dx = -600, 0

    start_time = time.time()

    # image_resize_without_mask(input_image, output_image, dy, dx)
    # image_resize_with_mask(input_image, output_image, dy, dx, input_mask)
    object_removal(input_image, output_image, input_mask)

    elapsed_time = time.time() - start_time
    print("Time elapsed: {}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))






