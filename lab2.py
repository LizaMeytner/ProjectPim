import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")
        self.root.geometry("1200x800")

        self.original_image = None
        self.processed_image = None
        self.current_image = None

        self.setup_ui()

    def setup_ui(self):
        # Основные фреймы
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - управление
        control_frame = ttk.LabelFrame(main_frame, text="Управление", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Правая панель - изображения
        image_frame = ttk.LabelFrame(main_frame, text="Изображения")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Загрузка и сохранение
        io_frame = ttk.LabelFrame(control_frame, text="Файловые операции")
        io_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(io_frame, text="Загрузить изображение",
                   command=self.load_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(io_frame, text="Сохранить результат",
                   command=self.save_image).pack(fill=tk.X, padx=5, pady=2)

        # Морфологические операции
        morph_frame = ttk.LabelFrame(control_frame, text="Морфологические операции")
        morph_frame.pack(fill=tk.X, padx=5, pady=5)

        # Структурный элемент
        ttk.Label(morph_frame, text="Структурный элемент:").pack(anchor=tk.W)
        self.kernel_size = tk.StringVar(value="3")
        kernel_frame = ttk.Frame(morph_frame)
        kernel_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(kernel_frame, textvariable=self.kernel_size, width=10).pack(side=tk.LEFT)
        ttk.Button(kernel_frame, text="Применить",
                   command=self.apply_custom_kernel).pack(side=tk.RIGHT)

        # Кнопки морфологических операций
        operations = [
            ("Эрозия", self.apply_erosion),
            ("Дилатация", self.apply_dilation),
            ("Открытие", self.apply_opening),
            ("Закрытие", self.apply_closing),
            ("Градиент", self.apply_gradient),
            ("Цилиндр", self.apply_tophat),
            ("Чёрная шляпа", self.apply_blackhat)
        ]

        for text, command in operations:
            ttk.Button(morph_frame, text=text, command=command).pack(fill=tk.X, padx=5, pady=1)

        # Фильтры и эффекты
        filter_frame = ttk.LabelFrame(control_frame, text="Фильтры и эффекты")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)

        filters = [
            ("Повышение резкости", self.sharpen_image),
            ("Размытие в движении", self.motion_blur),
            ("Тиснение", self.emboss_effect),
            ("Медианный фильтр", self.median_filter)
        ]

        for text, command in filters:
            ttk.Button(filter_frame, text=text, command=command).pack(fill=tk.X, padx=5, pady=1)

        # Пользовательский фильтр
        custom_frame = ttk.LabelFrame(control_frame, text="Пользовательский фильтр")
        custom_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(custom_frame, text="Матрица 3x3 (через запятую):").pack(anchor=tk.W)
        self.custom_filter = tk.StringVar(value="0,-1,0,-1,5,-1,0,-1,0")
        ttk.Entry(custom_frame, textvariable=self.custom_filter, width=30).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(custom_frame, text="Применить фильтр",
                   command=self.apply_custom_filter).pack(fill=tk.X, padx=5, pady=2)

        # Область отображения изображений
        self.setup_image_display(image_frame)

    def setup_image_display(self, parent):
        # Фрейм для изображений
        images_frame = ttk.Frame(parent)
        images_frame.pack(fill=tk.BOTH, expand=True)

        # Оригинальное изображение
        orig_frame = ttk.LabelFrame(images_frame, text="Оригинальное изображение")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.original_canvas = tk.Canvas(orig_frame, bg='white', width=400, height=400)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Обработанное изображение
        proc_frame = ttk.LabelFrame(images_frame, text="Обработанное изображение")
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.processed_canvas = tk.Canvas(proc_frame, bg='white', width=400, height=400)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.current_image = self.original_image.copy()
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    save_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file_path, save_image)
                    messagebox.showinfo("Успех", "Изображение сохранено")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {str(e)}")
        else:
            messagebox.showwarning("Предупреждение", "Нет обработанного изображения для сохранения")

    def display_images(self):
        if self.original_image is not None:
            # Отображение оригинального изображения
            orig_display = self.resize_image(self.original_image, 400, 400)
            self.original_photo = ImageTk.PhotoImage(orig_display)
            self.original_canvas.delete("all")
            self.original_canvas.create_image(200, 200, image=self.original_photo, anchor=tk.CENTER)

            # Отображение обработанного изображения
            if self.processed_image is not None:
                proc_display = self.resize_image(self.processed_image, 400, 400)
                self.processed_photo = ImageTk.PhotoImage(proc_display)
                self.processed_canvas.delete("all")
                self.processed_canvas.create_image(200, 200, image=self.processed_photo, anchor=tk.CENTER)

    def resize_image(self, image, max_width, max_height):
        pil_image = Image.fromarray(image)
        pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return pil_image

    def get_kernel(self, size=3):
        try:
            size = int(self.kernel_size.get())
            return np.ones((size, size), np.uint8)
        except:
            return np.ones((3, 3), np.uint8)

    def apply_erosion(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                # Для цветного изображения применяем к каждому каналу
                channels = []
                for i in range(3):
                    channel = cv2.erode(self.current_image[:, :, i], kernel, iterations=1)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.erode(self.current_image, kernel, iterations=1)
            self.display_images()

    def apply_dilation(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                channels = []
                for i in range(3):
                    channel = cv2.dilate(self.current_image[:, :, i], kernel, iterations=1)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.dilate(self.current_image, kernel, iterations=1)
            self.display_images()

    def apply_opening(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                channels = []
                for i in range(3):
                    channel = cv2.morphologyEx(self.current_image[:, :, i], cv2.MORPH_OPEN, kernel)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, kernel)
            self.display_images()

    def apply_closing(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                channels = []
                for i in range(3):
                    channel = cv2.morphologyEx(self.current_image[:, :, i], cv2.MORPH_CLOSE, kernel)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, kernel)
            self.display_images()

    def apply_gradient(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                channels = []
                for i in range(3):
                    channel = cv2.morphologyEx(self.current_image[:, :, i], cv2.MORPH_GRADIENT, kernel)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.morphologyEx(self.current_image, cv2.MORPH_GRADIENT, kernel)
            self.display_images()

    def apply_tophat(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                channels = []
                for i in range(3):
                    channel = cv2.morphologyEx(self.current_image[:, :, i], cv2.MORPH_TOPHAT, kernel)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.morphologyEx(self.current_image, cv2.MORPH_TOPHAT, kernel)
            self.display_images()

    def apply_blackhat(self):
        if self.current_image is not None:
            kernel = self.get_kernel()
            if len(self.current_image.shape) == 3:
                channels = []
                for i in range(3):
                    channel = cv2.morphologyEx(self.current_image[:, :, i], cv2.MORPH_BLACKHAT, kernel)
                    channels.append(channel)
                self.processed_image = np.stack(channels, axis=2)
            else:
                self.processed_image = cv2.morphologyEx(self.current_image, cv2.MORPH_BLACKHAT, kernel)
            self.display_images()

    def sharpen_image(self):
        if self.current_image is not None:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
            self.display_images()

    def motion_blur(self):
        if self.current_image is not None:
            size = 15
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size)
            kernel = kernel / size
            self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
            self.display_images()

    def emboss_effect(self):
        if self.current_image is not None:
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
            self.display_images()

    def median_filter(self):
        if self.current_image is not None:
            self.processed_image = cv2.medianBlur(self.current_image, 5)
            self.display_images()

    def apply_custom_filter(self):
        if self.current_image is not None:
            try:
                kernel_values = [float(x.strip()) for x in self.custom_filter.get().split(',')]
                if len(kernel_values) == 9:
                    kernel = np.array(kernel_values).reshape(3, 3)
                    self.processed_image = cv2.filter2D(self.current_image, -1, kernel)
                    self.display_images()
                else:
                    messagebox.showerror("Ошибка", "Введите 9 значений для матрицы 3x3")
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректные значения в матрице")

    def apply_custom_kernel(self):
        # Обновляем ядро для морфологических операций
        messagebox.showinfo("Информация", "Размер структурного элемента обновлен")


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()