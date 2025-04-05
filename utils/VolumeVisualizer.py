import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import imageio

class VolumeVisualizer:

    @staticmethod
    def show_slices(volume: np.array ,cmap: str='gray'):
        """
        Show the volume slice-by-slice using matplotlib
        :param volume: image volume
        :param cmap: colormap name
        :return:
        """
        num_slices = volume.shape[2]
        for i in range(num_slices):
            plt.imshow(volume[:, :, i], cmap=cmap)
            plt.title(f"Slice {i + 1}/{num_slices}")
            plt.axis("off")
            plt.pause(0.1)
        plt.close()

    @staticmethod
    def save_slices_as_png(volume: np.array, output_dir: str, indices=None, cmap: str='gray'):
        """
        Save slices as PNG images. Optionally select specific indices.
        :param volume: image volume
        :param output_dir: directory where to save png files
        :param indices: list of the specific indicies over Z-axis
        :param cmap: colormap name
        :return:
        """
        os.makedirs(output_dir, exist_ok=True)
        indices = indices if indices is not None else range(volume.shape[2])
        for i in indices:
            plt.imsave(os.path.join(output_dir, f"slice_{i:03d}.png"), volume[:, :, i], cmap=cmap)

    @staticmethod
    def generate_video(volume: np.array, output_path: str, fps: int = 10):
        """
        Create an MP4 video from all slices.
        :param volume:
        :param output_path:
        :param fps:
        :return:
        """
        h, w = volume.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)

        for i in range(volume.shape[2]):
            frame = (volume[:, :, i] * 255).clip(0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)
        video_writer.release()

    @staticmethod
    def generate_gif(volume: np.array, output_path: str, indices: list, step_ms: int = 100, cmap: str='gray'):
        """
        Create a GIF from specific slice indices.
        :param volume:
        :param output_path:
        :param indices:
        :param step_ms:
        :param cmap:
        :return:
        """
        images = []
        for i in indices:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(volume[:, :, i], cmap=cmap)
            ax.axis('off')
            fig.canvas.draw()

            # Convert canvas to image array
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)

        imageio.mimsave(output_path, images, duration=step_ms / 1000.0)
