# https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/demo_annotation_box.html

from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import argparse


def main():
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--path', type=str, default='../results/001_MANet_aniso_x4_test_stage1/toy_dataset/npz/toy1.npz',
                      help='path to  kernel estimation npz file')
    args = prog.parse_args()

    data = np.load(args.path)

    sr_img = data['sr_img'][:, :, [2, 1, 0]]
    est_ker_sv = data['est_ker_sv']
    gt_k_np = data['gt_ker']

    # create figure and plot scatter
    fig = plt.figure()

    if gt_k_np.sum() == 0:
        ax = fig.add_subplot(111)
        im = ax.imshow(sr_img)
    else:

        ax = fig.add_subplot(121)
        im = ax.imshow(gt_k_np, vmin=gt_k_np.min(), vmax=gt_k_np.max())
        plt.colorbar(im, ax=ax)
        ax.set_title('GT kernel')

        ax = fig.add_subplot(122)
        im = ax.imshow(sr_img)
        ax.set_title('View kernel estimation\n by hovering the cursor')

    # create the annotations box
    image = OffsetImage(np.random.rand(13, 13), zoom=5)
    xybox = (100., 100.)
    ab = AnnotationBbox(image, (0, 0), xybox=xybox, xycoords='data',
                        boxcoords="offset points", pad=0.3, arrowprops=dict(arrowstyle="->"))

    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if im.contains(event)[0]:

            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            ab.xybox = (-150, 0)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (int(event.xdata), int(event.ydata))
            # set the image corresponding to that point
            data = est_ker_sv[int(event.xdata) + int(event.ydata) * sr_img.shape[1], :, :]
            data = data / data.max()
            image.set_data(data)


        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()


if __name__ == '__main__':
    main()
