import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")
        self.root.geometry("1200x800")

        self.original_image = None
        self.current_image = None
        self.image_path = None

        self.setup_ui()

    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - управление
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Правая панель - изображение и гистограмма
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Верхняя часть правой панели - изображение
        self.image_frame = ttk.Frame(right_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)

        # Нижняя часть правой панели - гистограмма
        self.hist_frame = ttk.Frame(right_frame)
        self.hist_frame.pack(fill=tk.BOTH, expand=True)

        # Элементы управления
        self.create_controls(left_frame)

    def create_controls(self, parent):
        # Кнопка загрузки
        ttk.Button(parent, text="Загрузить изображение",
                   command=self.load_image).pack(pady=5, fill=tk.X)

        # Кнопка сохранения
        ttk.Button(parent, text="Сохранить изображение",
                   command=self.save_image).pack(pady=5, fill=tk.X)

        # Информация об изображении
        info_frame = ttk.LabelFrame(parent, text="Информация об изображении")
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = tk.Text(info_frame, height=10, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Преобразования
        transform_frame = ttk.LabelFrame(parent, text="Преобразования")
        transform_frame.pack(fill=tk.X, pady=5)

        ttk.Button(transform_frame, text="В градации серого",
                   command=self.convert_to_grayscale).pack(pady=2, fill=tk.X)

        ttk.Button(transform_frame, text="Поворот на 90°",
                   command=self.rotate_90).pack(pady=2, fill=tk.X)

        # Коррекция яркости
        correction_frame = ttk.LabelFrame(parent, text="Коррекция изображения")
        correction_frame.pack(fill=tk.X, pady=5)

        # Яркость
        ttk.Label(correction_frame, text="Яркость:").pack(anchor=tk.W)
        self.brightness_var = tk.DoubleVar(value=1.0)
        brightness_scale = ttk.Scale(correction_frame, from_=0.1, to=2.0,
                                     variable=self.brightness_var, orient=tk.HORIZONTAL)
        brightness_scale.pack(fill=tk.X)
        ttk.Button(correction_frame, text="Применить яркость",
                   command=self.apply_brightness).pack(pady=2, fill=tk.X)

        # Контрастность
        ttk.Label(correction_frame, text="Контрастность:").pack(anchor=tk.W)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(correction_frame, from_=0.1, to=2.0,
                                   variable=self.contrast_var, orient=tk.HORIZONTAL)
        contrast_scale.pack(fill=tk.X)
        ttk.Button(correction_frame, text="Применить контраст",
                   command=self.apply_contrast).pack(pady=2, fill=tk.X)

        # Насыщенность
        ttk.Label(correction_frame, text="Насыщенность:").pack(anchor=tk.W)
        self.saturation_var = tk.DoubleVar(value=1.0)
        saturation_scale = ttk.Scale(correction_frame, from_=0.0, to=2.0,
                                     variable=self.saturation_var, orient=tk.HORIZONTAL)
        saturation_scale.pack(fill=tk.X)
        ttk.Button(correction_frame, text="Применить насыщенность",
                   command=self.apply_saturation).pack(pady=2, fill=tk.X)

        # Гистограмма
        hist_frame = ttk.LabelFrame(parent, text="Гистограмма")
        hist_frame.pack(fill=tk.X, pady=5)

        ttk.Button(hist_frame, text="Построить гистограмму",
                   command=self.show_histogram).pack(pady=2, fill=tk.X)

        self.hist_channel = tk.StringVar(value="RGB")
        ttk.Radiobutton(hist_frame, text="RGB", variable=self.hist_channel,
                        value="RGB").pack(anchor=tk.W)
        ttk.Radiobutton(hist_frame, text="R", variable=self.hist_channel,
                        value="R").pack(anchor=tk.W)
        ttk.Radiobutton(hist_frame, text="G", variable=self.hist_channel,
                        value="G").pack(anchor=tk.W)
        ttk.Radiobutton(hist_frame, text="B", variable=self.hist_channel,
                        value="B").pack(anchor=tk.W)

        # Линейная коррекция
        ttk.Button(parent, text="Линейная коррекция (Ч/Б)",
                   command=self.linear_correction).pack(pady=2, fill=tk.X)

        ttk.Button(parent, text="Нелинейная коррекция (Ч/Б)",
                   command=self.nonlinear_correction).pack(pady=2, fill=tk.X)

        # Сброс
        ttk.Button(parent, text="Сбросить изменения",
                   command=self.reset_image).pack(pady=5, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                self.display_image()
                self.show_image_info()
                self.show_histogram()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def display_image(self):
        if not self.current_image:
            return

        # Очищаем фрейм
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # Масштабируем изображение для отображения
        display_image = self.current_image.copy()
        width, height = display_image.size

        # Ограничиваем размер для отображения
        max_size = 400
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)

        # Конвертируем для tkinter
        photo = ImageTk.PhotoImage(display_image)

        # Создаем label для изображения
        image_label = ttk.Label(self.image_frame, image=photo)
        image_label.image = photo  # сохраняем ссылку
        image_label.pack(pady=10)

    def show_image_info(self):
        if not self.original_image:
            return

        self.info_text.delete(1.0, tk.END)

        # Основная информация
        info = f"=== ИНФОРМАЦИЯ ОБ ИЗОБРАЖЕНИИ ===\n\n"
        info += f"Размер файла: {os.path.getsize(self.image_path)} байт\n"
        info += f"Разрешение: {self.original_image.size[0]} x {self.original_image.size[1]}\n"
        info += f"Формат: {self.original_image.format}\n"
        info += f"Цветовая модель: {self.original_image.mode}\n"

        # Глубина цвета
        if self.original_image.mode == 'L':
            info += f"Глубина цвета: 8 бит (256 градаций серого)\n"
        elif self.original_image.mode == 'RGB':
            info += f"Глубина цвета: 24 бита (8 бит на канал)\n"
        elif self.original_image.mode == 'RGBA':
            info += f"Глубина цвета: 32 бита (8 бит на канал + альфа)\n"
        else:
            info += f"Глубина цвета: информация недоступна\n"

        # EXIF информация
        try:
            exif_data = self.original_image._getexif()
            if exif_data:
                info += f"\n=== EXIF ДАННЫЕ ===\n"

                # Теги EXIF
                exif_tags = {
                    271: "Производитель камеры",
                    272: "Модель камеры",
                    306: "Дата и время",
                    36867: "Дата съемки",
                    36868: "Дата изменения",
                    37377: "Выдержка",
                    37378: "Диафрагма",
                    37379: "ISO",
                    37380: "Вспышка",
                    37383: "Фокусное расстояние",
                    274: "Ориентация"
                }

                count = 0
                for tag, value in exif_data.items():
                    tag_name = exif_tags.get(tag, f"Тег {tag}")
                    if tag in exif_tags and count < 5:
                        info += f"{tag_name}: {value}\n"
                        count += 1
            else:
                info += f"\nEXIF данные: недоступны\n"
        except:
            info += f"\nEXIF данные: ошибка чтения\n"

        # Дополнительная информация
        info += f"\n=== ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ===\n"
        info += f"Загружено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        info += f"Путь к файлу: {self.image_path}\n"

        self.info_text.insert(1.0, info)

    def convert_to_grayscale(self):
        if self.current_image:
            self.current_image = self.current_image.convert('L')
            self.display_image()
            self.show_histogram()

    def rotate_90(self):
        if self.current_image:
            self.current_image = self.current_image.rotate(-90, expand=True)
            self.display_image()

    def apply_brightness(self):
        if self.current_image:
            enhancer = ImageEnhance.Brightness(self.current_image)
            self.current_image = enhancer.enhance(self.brightness_var.get())
            self.display_image()
            self.show_histogram()

    def apply_contrast(self):
        if self.current_image:
            enhancer = ImageEnhance.Contrast(self.current_image)
            self.current_image = enhancer.enhance(self.contrast_var.get())
            self.display_image()
            self.show_histogram()

    def apply_saturation(self):
        if self.current_image and self.current_image.mode != 'L':
            enhancer = ImageEnhance.Color(self.current_image)
            self.current_image = enhancer.enhance(self.saturation_var.get())
            self.display_image()
            self.show_histogram()

    def show_histogram(self):
        if not self.current_image:
            return

        # Очищаем фрейм гистограммы
        for widget in self.hist_frame.winfo_children():
            widget.destroy()

        # Создаем фигуру matplotlib
        fig, ax = plt.subplots(figsize=(8, 3))

        # Конвертируем в RGB для гистограммы
        if self.current_image.mode == 'L':
            # Для grayscale
            gray_array = list(self.current_image.getdata())
            ax.hist(gray_array, bins=256, range=(0, 255), color='black', alpha=0.7)
            ax.set_title('Гистограмма (Grayscale)')
        else:
            # Для цветного
            rgb_image = self.current_image.convert('RGB')
            r, g, b = rgb_image.split()

            channel = self.hist_channel.get()
            if channel == 'RGB':
                ax.hist(list(r.getdata()), bins=256, range=(0, 255), color='red', alpha=0.5, label='R')
                ax.hist(list(g.getdata()), bins=256, range=(0, 255), color='green', alpha=0.5, label='G')
                ax.hist(list(b.getdata()), bins=256, range=(0, 255), color='blue', alpha=0.5, label='B')
                ax.legend()
                ax.set_title('Гистограмма (RGB)')
            elif channel == 'R':
                ax.hist(list(r.getdata()), bins=256, range=(0, 255), color='red', alpha=0.7)
                ax.set_title('Гистограмма (Red канал)')
            elif channel == 'G':
                ax.hist(list(g.getdata()), bins=256, range=(0, 255), color='green', alpha=0.7)
                ax.set_title('Гистограмма (Green канал)')
            elif channel == 'B':
                ax.hist(list(b.getdata()), bins=256, range=(0, 255), color='blue', alpha=0.7)
                ax.set_title('Гистограмма (Blue канал)')

        ax.set_xlabel('Интенсивность')
        ax.set_ylabel('Частота')
        ax.grid(True, alpha=0.3)

        # Встраиваем в tkinter
        canvas = FigureCanvasTkAgg(fig, self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def linear_correction(self):
        if self.current_image:
            # Конвертируем в grayscale если нужно
            if self.current_image.mode != 'L':
                temp_image = self.current_image.convert('L')
            else:
                temp_image = self.current_image

            # Линейное растяжение гистограммы
            self.current_image = ImageOps.autocontrast(temp_image)
            self.display_image()
            self.show_histogram()

    def nonlinear_correction(self):
        if self.current_image:
            # Конвертируем в grayscale если нужно
            if self.current_image.mode != 'L':
                temp_image = self.current_image.convert('L')
            else:
                temp_image = self.current_image

            # Нелинейная коррекция (гамма-коррекция)
            import numpy as np
            array = np.array(temp_image)
            # Гамма = 0.5 для осветления теней
            corrected_array = np.power(array / 255.0, 0.5) * 255
            self.current_image = Image.fromarray(corrected_array.astype('uint8'))
            self.display_image()
            self.show_histogram()

    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.brightness_var.set(1.0)
            self.contrast_var.set(1.0)
            self.saturation_var.set(1.0)
            self.display_image()
            self.show_histogram()

    def save_image(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("BMP files", "*.bmp"),
                    ("All files", "*.*")
                ]
            )

            if file_path:
                try:
                    self.current_image.save(file_path)
                    messagebox.showinfo("Успех", "Изображение успешно сохранено!")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {str(e)}")


def main():
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()