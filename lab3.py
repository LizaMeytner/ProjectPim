import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображений")
        self.root.geometry("1200x800")

        self.original_image = None
        self.processed_image = None
        self.current_image = None

        self.setup_ui()

    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Панель управления
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Кнопки загрузки и обработки
        ttk.Button(control_frame, text="Загрузить изображение",
                   command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Детектор Кэнни",
                   command=self.apply_canny).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Оператор Робертса",
                   command=self.apply_roberts).pack(side=tk.LEFT, padx=5)

        # Кластеризация
        cluster_frame = ttk.Frame(control_frame)
        cluster_frame.pack(side=tk.LEFT, padx=10)

        ttk.Label(cluster_frame, text="Кластеры:").pack(side=tk.LEFT)
        self.cluster_var = tk.IntVar(value=5)
        cluster_spinbox = ttk.Spinbox(cluster_frame, from_=2, to=20,
                                      textvariable=self.cluster_var, width=5)
        cluster_spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Button(cluster_frame, text="Сегментация",
                   command=self.apply_clustering).pack(side=tk.LEFT, padx=5)

        # Ключевые точки
        ttk.Button(control_frame, text="Ключевые точки (SIFT)",
                   command=self.detect_keypoints).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Найти номер",
                   command=self.detect_license_plate).pack(side=tk.LEFT, padx=5)

        # Область отображения изображений
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)

        # Исходное изображение
        self.original_frame = ttk.LabelFrame(display_frame, text="Исходное изображение")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Обработанное изображение
        self.processed_frame = ttk.LabelFrame(display_frame, text="Обработанное изображение")
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.processed_label = ttk.Label(self.processed_frame)
        self.processed_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                # Проверка существования файла
                if not os.path.exists(file_path):
                    messagebox.showerror("Ошибка", f"Файл не существует: {file_path}")
                    return

                # Чтение изображения с проверкой
                image = cv2.imread(file_path)
                if image is None:
                    # Попробуем альтернативный способ через PIL
                    try:
                        pil_image = Image.open(file_path)
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    except Exception as pil_error:
                        messagebox.showerror("Ошибка",
                                             f"Не удалось загрузить изображение:\n"
                                             f"OpenCV error: Файл поврежден или неподдерживаемый формат\n"
                                             f"PIL error: {str(pil_error)}")
                        return

                # Конвертация цветового пространства
                self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = self.original_image.copy()
                self.processed_image = None

                # Отображение
                self.display_images()
                messagebox.showinfo("Успех", f"Изображение загружено: {os.path.basename(file_path)}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")

    def display_images(self):
        if self.original_image is not None:
            # Масштабирование для отображения
            display_original = self.resize_image(self.original_image, 400)
            self.original_photo = ImageTk.PhotoImage(display_original)
            self.original_label.configure(image=self.original_photo)

        if self.processed_image is not None:
            display_processed = self.resize_image(self.processed_image, 400)
            self.processed_photo = ImageTk.PhotoImage(display_processed)
            self.processed_label.configure(image=self.processed_photo)

    def resize_image(self, image, max_size):
        h, w = image.shape[:2]
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)

        img_pil = Image.fromarray(image)
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return img_resized

    def apply_canny(self):
        if self.original_image is not None:
            try:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                self.display_images()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка в детекторе Кэнни: {str(e)}")

    def apply_roberts(self):
        if self.original_image is not None:
            try:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

                # Оператор Робертса
                kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
                kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

                roberts_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
                roberts_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

                roberts = np.sqrt(roberts_x ** 2 + roberts_y ** 2)
                roberts = np.uint8(roberts / roberts.max() * 255)

                self.processed_image = cv2.cvtColor(roberts, cv2.COLOR_GRAY2RGB)
                self.display_images()

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка в операторе Робертса: {str(e)}")

    def apply_clustering(self):
        if self.original_image is not None:
            try:
                n_clusters = self.cluster_var.get()

                # Подготовка данных для кластеризации
                image_reshaped = self.original_image.reshape(-1, 3)

                # Применение K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(image_reshaped)

                # Создание сегментированного изображения
                segmented = kmeans.cluster_centers_[labels].reshape(self.original_image.shape)
                segmented = segmented.astype(np.uint8)

                self.processed_image = segmented
                self.display_images()

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка в кластеризации: {str(e)}")

    def detect_keypoints(self):
        if self.original_image is not None:
            try:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

                # Детектор SIFT
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray, None)

                # Рисование ключевых точек
                result_image = self.original_image.copy()
                result_image = cv2.drawKeypoints(result_image, keypoints, None,
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                self.processed_image = result_image
                self.display_images()

                messagebox.showinfo("Информация",
                                    f"Найдено ключевых точек: {len(keypoints)}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка в детектировании ключевых точек: {str(e)}")

    def detect_license_plate(self):
        if self.original_image is not None:
            try:
                # Создаем копию оригинального изображения для результата
                result_image = self.original_image.copy()
                h, w = result_image.shape[:2]

                # Масштабируем изображение для лучшей обработки (если слишком большое)
                scale_factor = 1.0
                if w > 1200:
                    scale_factor = 1200.0 / w
                    new_width = 1200
                    new_height = int(h * scale_factor)
                    working_image = cv2.resize(self.original_image, (new_width, new_height))
                else:
                    working_image = self.original_image.copy()

                # Конвертируем в разные цветовые пространства
                gray = cv2.cvtColor(working_image, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(working_image, cv2.COLOR_RGB2HSV)
                lab = cv2.cvtColor(working_image, cv2.COLOR_RGB2LAB)

                # Метод 1: Поиск белых областей (типично для российских номеров)
                # Белый в HSV: V > 200, S < 50
                white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))

                # Метод 2: Поиск светлых областей в LAB (L-канал)
                _, light_mask = cv2.threshold(lab[:, :, 0], 200, 255, cv2.THRESH_BINARY)

                # Метод 3: Поиск контрастных областей через градиенты
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
                gradient_magnitude = np.uint8(255 * gradient_magnitude / gradient_magnitude.max())
                _, gradient_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

                # Комбинируем маски
                combined_mask = cv2.bitwise_or(white_mask, light_mask)
                combined_mask = cv2.bitwise_or(combined_mask, gradient_mask)

                # Морфологические операции для улучшения маски
                kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
                kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

                # Закрытие для соединения близких областей
                morph = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_rect)
                # Открытие для удаления мелких шумов
                morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_square)
                # Дилатация для усиления областей
                morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel_rect)

                # Поиск контуров
                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                license_plate_found = False
                potential_plates = []

                for contour in contours:
                    # Получаем ограничивающий прямоугольник
                    x, y, rect_w, rect_h = cv2.boundingRect(contour)

                    # Пропускаем слишком маленькие области
                    if rect_w < 40 or rect_h < 10:
                        continue

                    # Вычисляем характеристики
                    aspect_ratio = rect_w / rect_h
                    area = rect_w * rect_h
                    contour_area = cv2.contourArea(contour)
                    extent = contour_area / area if area > 0 else 0

                    # Эвристики для российских номеров
                    is_potential_plate = (
                        # Соотношение сторон (российские номера ~ 520x112 мм = 4.64)
                            (3.0 < aspect_ratio < 6.0) and
                            # Достаточная площадь
                            (area > 500) and
                            # Форма должна быть достаточно заполнена
                            (extent > 0.4) and
                            # Размеры в разумных пределах
                            (rect_w > 60) and (rect_h > 15)
                    )

                    if is_potential_plate:
                        # Добавляем отступы вокруг предполагаемого номера
                        padding_x = int(rect_w * 0.1)
                        padding_y = int(rect_h * 0.1)
                        x1 = max(0, x - padding_x)
                        y1 = max(0, y - padding_y)
                        x2 = min(working_image.shape[1], x + rect_w + padding_x)
                        y2 = min(working_image.shape[0], y + rect_h + padding_y)

                        # Вычисляем score для региона (на основе равномерности цвета и текстуры)
                        plate_region = working_image[y1:y2, x1:x2]
                        if plate_region.size == 0:
                            continue

                        # Анализ региона на текстуру (номера обычно имеют текст)
                        plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_RGB2GRAY)
                        plate_std = np.std(plate_gray)

                        # Регионы с высокой вариацией (текст) получают высокий score
                        texture_score = min(plate_std / 50.0, 1.0)

                        potential_plates.append({
                            'coords': (x1, y1, x2, y2),
                            'score': texture_score,
                            'aspect_ratio': aspect_ratio,
                            'area': area
                        })

                # Сортируем по score и выбираем лучшие кандидаты
                potential_plates.sort(key=lambda x: x['score'], reverse=True)

                # Отображаем лучшие кандидаты (до 3)
                for i, plate in enumerate(potential_plates[:3]):
                    x1, y1, x2, y2 = plate['coords']

                    # Масштабируем координаты обратно если изображение было уменьшено
                    if scale_factor != 1.0:
                        x1 = int(x1 / scale_factor)
                        y1 = int(y1 / scale_factor)
                        x2 = int(x2 / scale_factor)
                        y2 = int(y2 / scale_factor)

                    # Рисуем прямоугольник разными цветами в зависимости от score
                    color = (0, 255, 0) if i == 0 else (255, 255, 0)  # Зеленый для лучшего, желтый для остальных
                    thickness = 3 if i == 0 else 2

                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(result_image, f'Plate {i + 1}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Вырезаем и сохраняем лучший номер
                    if i == 0:
                        license_plate_region = self.original_image[y1:y2, x1:x2]
                        if license_plate_region.size > 0:
                            # Улучшаем контраст вырезанного номера
                            license_plate_gray = cv2.cvtColor(license_plate_region, cv2.COLOR_RGB2GRAY)
                            license_plate_enhanced = cv2.equalizeHist(license_plate_gray)

                            # Сохраняем номер
                            cv2.imwrite('detected_license_plate.png', license_plate_enhanced)

                    license_plate_found = True

                # Если не нашли по маскам, пробуем альтернативный метод - поиск прямоугольников
                if not license_plate_found:
                    # Используем детектор ребер
                    edges = cv2.Canny(gray, 50, 150)

                    # Ищем линии
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                            minLineLength=30, maxLineGap=10)

                    if lines is not None:
                        # Группируем линии для поиска прямоугольников
                        horizontal_lines = []
                        vertical_lines = []

                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                            if abs(angle) < 30:  # Горизонтальные линии
                                horizontal_lines.append(line[0])
                            elif abs(angle - 90) < 30:  # Вертикальные линии
                                vertical_lines.append(line[0])

                        # Простой поиск пересечений для формирования прямоугольников
                        # (здесь можно добавить более сложную логику группировки)

                self.processed_image = result_image
                self.display_images()

                if not license_plate_found:
                    messagebox.showwarning("Предупреждение",
                                           "Номерной знак не найден. Попробуйте:\n"
                                           "- Более четкое изображение\n"
                                           "- Прямой угол съемки\n"
                                           "- Хорошее освещение")
                else:
                    best_plate = potential_plates[0] if potential_plates else None
                    messagebox.showinfo("Успех",
                                        f"Найдено кандидатов: {len(potential_plates)}\n"
                                        f"Лучший score: {best_plate['score']:.2f}\n"
                                        f"Номер сохранен как 'detected_license_plate.png'")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка в детектировании номера: {str(e)}")
        else:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")


def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()