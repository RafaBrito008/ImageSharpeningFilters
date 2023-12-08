import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft2, fftshift
import os


class SmoothingFilters:
    @staticmethod
    def apply_average_filter(image, kernel_size):
        """
        Aplica un filtro promedio a la imagen proporcionada utilizando un kernel cuadrado del tamaño especificado.
        El filtro promedio reemplaza cada píxel con el promedio de los píxeles de su vecindario.

        :param image: Imagen de entrada como una matriz NumPy.
        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :return: Imagen filtrada.
        """
        # Agrega píxeles alrededor de los bordes para permitir el cálculo del vecindario para los píxeles de borde.
        padded_image = cv2.copyMakeBorder(
            image,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            cv2.BORDER_REFLECT,
        )

        # Inicializa la imagen de salida con ceros, del mismo tamaño que la imagen de entrada.
        output_image = np.zeros_like(image)

        # Itera sobre cada píxel de la imagen.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extrae la región del vecindario del píxel actual basado en el tamaño del kernel.
                kernel_region = padded_image[i : i + kernel_size, j : j + kernel_size]
                # Calcula el promedio de los píxeles del vecindario y asigna el resultado al píxel correspondiente.
                output_image[i, j] = np.mean(kernel_region)

        return output_image

    @staticmethod
    def apply_median_filter(image, kernel_size):
        """
        Aplica un filtro de mediana a la imagen utilizando un kernel cuadrado del tamaño especificado.
        El filtro de mediana reemplaza cada píxel por la mediana de los píxeles en su vecindario.

        :param image: Imagen de entrada como una matriz NumPy.
        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :return: Imagen filtrada.
        """
        # Agrega píxeles alrededor de los bordes de la misma manera que en el filtro promedio.
        padded_image = cv2.copyMakeBorder(
            image,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            cv2.BORDER_REFLECT,
        )

        # Inicializa la imagen de salida con ceros.
        output_image = np.zeros_like(image)

        # Itera sobre cada píxel de la imagen.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extrae la región del vecindario del píxel actual.
                kernel_region = padded_image[i : i + kernel_size, j : j + kernel_size]
                # Calcula la mediana de los píxeles del vecindario y asigna el resultado al píxel correspondiente.
                output_image[i, j] = np.median(kernel_region)

        return output_image

    @staticmethod
    def generate_gaussian_kernel(kernel_size, sigma):
        """
        Genera un kernel gaussiano que se utilizará para el filtrado gaussiano.

        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :param sigma: Desviación estándar de la distribución gaussiana.
        :return: Kernel gaussiano como una matriz NumPy.
        """
        # Crea un rango de valores de acuerdo al tamaño del kernel para calcular el kernel gaussiano.
        ax = np.linspace(-(kernel_size - 1) / 2.0, (kernel_size - 1) / 2.0, kernel_size)
        # Crea una cuadrícula 2D de valores desde la matriz lineal.
        xx, yy = np.meshgrid(ax, ax)
        # Aplica la fórmula gaussiana a cada elemento de la cuadrícula.
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        # Normaliza el kernel para que la suma de todos sus elementos sea igual a 1.
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_gaussian_filter(image, kernel_size, sigma):
        """
        Aplica un filtro gaussiano a la imagen proporcionada.

        :param image: Imagen de entrada como una matriz NumPy.
        :param kernel_size: Tamaño del lado del kernel cuadrado.
        :param sigma: Desviación estándar del kernel gaussiano.
        :return: Imagen filtrada.
        """
        # Genera el kernel gaussiano utilizando la función anterior.
        kernel = SmoothingFilters.generate_gaussian_kernel(kernel_size, sigma)
        # Agrega píxeles alrededor de los bordes de la imagen.
        padded_image = cv2.copyMakeBorder(
            image,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            kernel_size // 2,
            cv2.BORDER_REFLECT,
        )

        # Inicializa la imagen de salida con ceros.
        output_image = np.zeros_like(image)

        # Aplica el kernel gaussiano a cada píxel de la imagen.
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extrae la región correspondiente al vecindario del píxel.
                kernel_region = padded_image[i : i + kernel_size, j : j + kernel_size]
                # Realiza la operación de convolución: multiplica el kernel por los píxeles del vecindario y suma los resultados.
                output_image[i, j] = np.sum(kernel_region * kernel)

        return output_image


class SharpeningFilters:
    @staticmethod
    def apply_convolution(image, kernel):
        """
        Aplica la convolución con un kernel dado sobre una imagen.
        """
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def apply_laplacian_filter(image):
        """
        Aplica un filtro Laplaciano para acentuar los bordes.
        """
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        return SharpeningFilters.apply_convolution(image, kernel)

    @staticmethod
    def apply_sobel_filter(image, direction='both'):
        """
        Aplica un filtro de Sobel para detectar bordes tanto en dirección horizontal como vertical.
        """
        if direction not in ['horizontal', 'vertical', 'both']:
            raise ValueError("direction debe ser 'horizontal', 'vertical' o 'both'.")

        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) if direction in ['horizontal', 'both'] else 0
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) if direction in ['vertical', 'both'] else 0

        if direction == 'both':
            grad = cv2.magnitude(grad_x, grad_y)
            return cv2.convertScaleAbs(grad)
        else:
            return cv2.convertScaleAbs(grad_x if direction == 'horizontal' else grad_y)

    @staticmethod
    def apply_prewitt_filter(image, direction='both'):
        """
        Aplica un filtro de Prewitt para detectar bordes tanto en dirección horizontal como vertical.
        """
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

        grad_x = SharpeningFilters.apply_convolution(image, prewitt_x) if direction in ['horizontal', 'both'] else 0
        grad_y = SharpeningFilters.apply_convolution(image, prewitt_y) if direction in ['vertical', 'both'] else 0

        if direction == 'both':
            grad = cv2.magnitude(grad_x, grad_y)
            return cv2.convertScaleAbs(grad)
        else:
            return cv2.convertScaleAbs(grad_x if direction == 'horizontal' else grad_y)

    @staticmethod
    def apply_roberts_filter(image):
        """
        Aplica un filtro de Roberts para detectar bordes diagonalmente.
        """
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        grad_x = SharpeningFilters.apply_convolution(image, roberts_x)
        grad_y = SharpeningFilters.apply_convolution(image, roberts_y)
        grad = cv2.magnitude(grad_x, grad_y)
        return cv2.convertScaleAbs(grad)



# Clase principal de la aplicación de procesamiento de imágenes
class ImageProcessorApp:
    def __init__(self, root):
        # Constructor de la clase
        self.root = root  # Referencia a la ventana principal de Tkinter
        self.root.title("Procesador de Imágenes")  # Título de la ventana
        self.image_path = None  # Ruta de la imagen a procesar
        self.setup_ui()  # Inicializar la interfaz de usuario

    def setup_ui(self):
        # Configuración de la interfaz de usuario
        self.frame = tk.Frame(self.root)  # Crear un marco en la ventana principal
        self.frame.pack(padx=10, pady=10)  # Empaquetar el marco con un poco de espacio

        # Botón para cargar imágenes
        self.button_load = tk.Button(
            self.frame, text="Cargar Imagen", command=self.load_image
        )
        self.button_load.pack(side=tk.TOP, pady=5)  # Posicionar el botón

        # Crear un lienzo para mostrar imágenes
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)  # Empaquetar el lienzo

        # Vincular el evento de cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Método para manejar el cierre de la ventana
        self.root.quit()  # Terminar el bucle principal
        self.root.destroy()  # Destruir la ventana

    def load_image(self):
        # Método para cargar una imagen
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")]
        )  # Mostrar un cuadro de diálogo para elegir un archivo
        if self.image_path:
            self.process_image()  # Si se selecciona una imagen, procesarla

    def save_image(self, image, image_name):
        """
        Guarda una imagen procesada en el disco duro.

        :param image: Imagen procesada a guardar.
        :param image_name: Nombre bajo el cual se guardará la imagen.
        """
        # Obtener el nombre base de la imagen original
        original_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Crear un nombre de carpeta con '_Processed' al final
        folder_name = f"{original_name}_Processed"

        # Verificar si la carpeta existe. Si no, crearla.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Construir la ruta completa del archivo donde se guardará la imagen
        filename = os.path.join(folder_name, f"{original_name}_{image_name}.jpg")

        # Guardar la imagen en el formato deseado
        cv2.imwrite(filename, image)

    def process_image(self):
        # Parámetros configurables para el procesamiento de la imagen
        noise_intensity = 25  # Intensidad del ruido a añadir
        filter_size = 3  # Tamaño del filtro gaussiano
        gaussian_sigma = 1.5  # Desviación estándar del filtro gaussiano

        # Cargar la imagen y convertirla a escala de grises
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Añadir ruido tipo 'sal y pimienta'
        noise = np.random.normal(0, noise_intensity, img.shape).astype(np.uint8)
        img_noisy = cv2.add(img, noise)

        # Aplicar filtro de suavizado (Gaussiano en este caso)
        img_smoothed = SmoothingFilters.apply_gaussian_filter(img_noisy, filter_size, gaussian_sigma)

        # Convertir la imagen a float32 para el procesamiento
        img_smoothed = img_smoothed.astype(np.float32)

        # Aplicar filtros de acentuado
        img_laplacian = SharpeningFilters.apply_laplacian_filter(img_smoothed)
        img_sobel = SharpeningFilters.apply_sobel_filter(img_smoothed)
        img_prewitt = SharpeningFilters.apply_prewitt_filter(img_smoothed)
        img_roberts = SharpeningFilters.apply_roberts_filter(img_smoothed)

        # Guardar las imágenes procesadas
        self.save_image(img_smoothed, "smoothed")
        self.save_image(img_laplacian, "laplacian_filtered")
        self.save_image(img_sobel, "sobel_filtered")
        self.save_image(img_prewitt, "prewitt_filtered")
        self.save_image(img_roberts, "roberts_filtered")

        # Configuración de la visualización de resultados
        fig, axs = plt.subplots(2, 3, figsize=(10, 7))

        # Mostrar las imágenes procesadas en los subplots
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        axs[0, 0].set_title("Imagen Original")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(img_smoothed, cmap="gray")
        axs[0, 1].set_title("Imagen Suavizada")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(img_laplacian, cmap="gray")
        axs[0, 2].set_title("Filtro Laplaciano")
        axs[0, 2].axis("off")

        axs[1, 0].imshow(img_sobel, cmap="gray")
        axs[1, 0].set_title("Filtro Sobel")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(img_prewitt, cmap="gray")
        axs[1, 1].set_title("Filtro Prewitt")
        axs[1, 1].axis("off")

        axs[1, 2].imshow(img_roberts, cmap="gray")
        axs[1, 2].set_title("Filtro Roberts")
        axs[1, 2].axis("off")

        # Ajustar etiquetas y mostrar figura en el lienzo de TkinterTkinter
        for ax in axs.flat:
            ax.label_outer()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()


# Bloque principal para ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
