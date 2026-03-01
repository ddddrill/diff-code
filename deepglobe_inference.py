"""
DeepGlobe Land Cover Classification - Local Inference
======================================================
Этот скрипт загружает обученную модель DeepLabV3+ и использует её для предсказания.

Использование:
    python deepglobe_inference.py --image path/to/image.jpg --model model_weights.pth
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ==================== КОНФИГУРАЦИЯ ====================

# Классы датасета DeepGlobe
CLASS_NAMES = {
    0: 'Urban',
    1: 'Agriculture', 
    2: 'Rangeland',
    3: 'Forest',
    4: 'Water',
    5: 'Barren',
    6: 'Unknown'
}

# Цвета для визуализации (RGB)
CLASS_COLORS = {
    0: (0, 255, 255),     # Urban - голубой
    1: (255, 255, 0),     # Agriculture - жёлтый
    2: (255, 0, 255),     # Rangeland - фиолетовый/розовый
    3: (0, 255, 0),       # Forest - зелёный
    4: (0, 0, 255),       # Water - синий
    5: (255, 255, 255),   # Barren - белый
    6: (0, 0, 0),         # Unknown - чёрный
}

# Русские названия классов
CLASS_NAMES_RU = {
    0: 'Городские территории',
    1: 'Сельхоз земли', 
    2: 'Пастбища',
    3: 'Леса',
    4: 'Вода',
    5: 'Пустыни/пески',
    6: 'Неизвестно'
}

# Параметры модели (должны совпадать с обучением)
DEFAULT_ENCODER = 'resnet50'
DEFAULT_IMG_SIZE = 512
NUM_CLASSES = 7


# ==================== ФУНКЦИИ ====================

def mask_to_rgb(mask):
    """Конвертация маски классов в RGB изображение."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        rgb[mask == class_id] = color
    
    return rgb


def rgb_to_mask(rgb_mask, threshold=128):
    """Конвертация RGB маски в маску классов."""
    rgb_mask = np.array(rgb_mask)
    rgb_mask = (rgb_mask > threshold).astype(np.uint8) * 255
    
    h, w = rgb_mask.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        color_bin = (np.array(color) > threshold).astype(np.uint8) * 255
        matches = np.all(rgb_mask == color_bin, axis=-1)
        mask[matches] = class_id
    
    return mask


class DeepGlobeModel:
    """Класс для загрузки и использования модели DeepLabV3+."""
    
    def __init__(self, model_path, encoder=DEFAULT_ENCODER, img_size=DEFAULT_IMG_SIZE, device=None):
        """
        Инициализация модели.
        
        Args:
            model_path: Путь к файлу весов (.pth)
            encoder: Название энкодера (resnet34, resnet50, etc.)
            img_size: Размер входного изображения
            device: Устройство ('cuda', 'cpu' или None для автоопределения)
        """
        self.encoder = encoder
        self.img_size = img_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Создаём модель (пробуем разные варианты названий)
        try:
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.encoder,
                encoder_weights=None,
                classes=NUM_CLASSES,
                activation=None,
            )
        except AttributeError:
            try:
                self.model = smp.DeepLabV3(
                    encoder_name=self.encoder,
                    encoder_weights=None,
                    classes=NUM_CLASSES,
                    activation=None,
                )
            except AttributeError:
                raise AttributeError("Не удалось создать модель. Проверьте версию segmentation_models_pytorch")
        
        # Загружаем веса
        self.load_weights(model_path)
        
        # Трансформации
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.model.eval()
    
    def load_weights(self, model_path):
        """Загрузка весов модели."""
        print(f"Loading weights from: {model_path}")
        
        # Используем weights_only=True для безопасности (для новых файлов)
        # или weights_only=False для совместимости со старыми чекпоинтами
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Проверяем формат чекпоинта
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'config' in checkpoint:
                    print(f"Model config: {checkpoint['config']}")
                if 'best_iou' in checkpoint:
                    print(f"Best IoU: {checkpoint['best_iou']:.4f}")
            else:
                # Простой словарь с весами
                self.model.load_state_dict(checkpoint)
        else:
            # Напрямую state_dict
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        print("✅ Model loaded successfully!")
    
    def preprocess(self, image):
        """
        Предобработка изображения.
        
        Args:
            image: PIL Image или numpy array (H, W, 3)
        
        Returns:
            torch tensor (1, 3, H, W)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ресайз
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Трансформации
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image, return_original_size=False):
        """
        Предсказание маски для изображения.
        
        Args:
            image: PIL Image, numpy array или путь к файлу
            return_original_size: Вернуть маску в оригинальном размере
        
        Returns:
            numpy array с маской классов (H, W)
        """
        # Загрузка изображения если передан путь
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            original_size = image.size  # (width, height)
        elif isinstance(image, Image.Image):
            original_size = image.size
            image = np.array(image)
        else:
            original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Предобработка
        image_tensor = self.preprocess(image)
        
        # Предсказание
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Возврат к оригинальному размеру
        if return_original_size:
            pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)
        
        return pred
    
    def predict_rgb(self, image, return_original_size=True):
        """
        Предсказание RGB маски.
        
        Args:
            image: PIL Image, numpy array или путь к файлу
            return_original_size: Вернуть маску в оригинальном размере
        
        Returns:
            numpy array (H, W, 3) с цветами классов
        """
        mask = self.predict(image, return_original_size=return_original_size)
        return mask_to_rgb(mask)
    
    def predict_with_confidence(self, image, return_original_size=True):
        """
        Предсказание с уверенностью (вероятностями).
        
        Args:
            image: PIL Image, numpy array или путь к файлу
            return_original_size: Вернуть в оригинальном размере
        
        Returns:
            pred: маска классов (H, W)
            confidence: карта уверенности (H, W)
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            original_size = image.size
        elif isinstance(image, Image.Image):
            original_size = image.size
            image = np.array(image)
        else:
            original_size = (image.shape[1], image.shape[0])
        
        image_tensor = self.preprocess(image)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, dim=1)
            pred = pred.squeeze().cpu().numpy()
            confidence = confidence.squeeze().cpu().numpy()
        
        if return_original_size:
            pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(confidence, original_size, interpolation=cv2.INTER_LINEAR)
        
        return pred, confidence
    
    def compute_class_areas(self, mask):
        """
        Подсчёт площади каждого класса в процентах.
        
        Args:
            mask: маска классов (H, W)
        
        Returns:
            dict: {название_класса: процент}
        """
        total = mask.size
        areas = {}
        for class_id in range(NUM_CLASSES):
            count = (mask == class_id).sum()
            percentage = count / total * 100
            if percentage > 0.01:
                areas[CLASS_NAMES[class_id]] = percentage
        return areas
    
    def visualize_prediction(self, image, save_path=None, show=True, use_russian=True):
        """
        Визуализация предсказания с легендой классов.
        
        Args:
            image: PIL Image, numpy array или путь к файлу
            save_path: Путь для сохранения результата
            show: Показать результат
            use_russian: Использовать русские названия классов
        """
        if isinstance(image, str):
            original_image = Image.open(image).convert('RGB')
            image_path = image
        else:
            original_image = image
            image_path = None
        
        # Предсказание
        pred = self.predict(original_image, return_original_size=True)
        pred_rgb = mask_to_rgb(pred)
        
        # Подсчёт площадей
        class_areas = self.compute_class_areas(pred)
        
        # Названия классов
        class_names = CLASS_NAMES_RU if use_russian else CLASS_NAMES
        
        # Создаём фигуру
        fig = plt.figure(figsize=(18, 6))
        
        # Создаём grid: 3 колонки для изображений + 1 для легенды
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.5])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])
        
        # Оригинальное изображение
        ax1.imshow(original_image)
        ax1.set_title('Исходное изображение' if use_russian else 'Original Image', fontsize=12)
        ax1.axis('off')
        
        # Предсказанная маска
        ax2.imshow(pred_rgb)
        ax2.set_title('Предсказанная маска' if use_russian else 'Prediction Mask', fontsize=12)
        ax2.axis('off')
        
        # Наложение
        original_np = np.array(original_image)
        overlay = cv2.addWeighted(original_np, 0.6, pred_rgb, 0.4, 0)
        ax3.imshow(overlay)
        ax3.set_title('Наложение' if use_russian else 'Overlay', fontsize=12)
        ax3.axis('off')
        
        # Легенда с цветами и площадями
        ax4.axis('off')
        ax4.set_title('Легенда' if use_russian else 'Legend', fontsize=12)
        
        # Создаём легенду
        legend_elements = []
        legend_text = []
        
        # Сортируем по убыванию площади
        sorted_classes = sorted(range(NUM_CLASSES), 
                               key=lambda x: class_areas.get(CLASS_NAMES[x], 0), 
                               reverse=True)
        
        y_pos = 0.95
        for class_id in sorted_classes:
            color = np.array(CLASS_COLORS[class_id]) / 255.0
            name = class_names[class_id]
            area = class_areas.get(CLASS_NAMES[class_id], 0)
            
            if area > 0.01:  # Показываем только если класс есть
                # Цветной квадрат
                rect = mpatches.Rectangle((0.05, y_pos - 0.03), 0.15, 0.05, 
                                          linewidth=1, edgecolor='black',
                                          facecolor=color, transform=ax4.transAxes)
                ax4.add_patch(rect)
                
                # Название и процент
                if use_russian:
                    text = f'{name}: {area:.1f}%'
                else:
                    text = f'{name}: {area:.1f}%'
                ax4.text(0.25, y_pos - 0.005, text, transform=ax4.transAxes,
                        fontsize=10, verticalalignment='center')
                
                y_pos -= 0.08
        
        # Информация о модели
        ax4.text(0.05, 0.05, f'Model: {self.encoder}\nSize: {self.img_size}px',
                transform=ax4.transAxes, fontsize=8, color='gray',
                verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Сохранено: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        # Печать распределения классов
        print("\n" + "="*50)
        print("Распределение классов:" if use_russian else "Class Distribution:")
        print("="*50)
        for name, pct in sorted(class_areas.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 2)
            print(f"{name:20s}: {pct:6.2f}% {bar}")
        print("="*50)
        
        return pred, class_areas
    
    def save_prediction(self, image_path, output_path):
        """
        Предсказание и сохранение маски.
        
        Args:
            image_path: путь к изображению
            output_path: путь для сохранения маски
        """
        pred_rgb = self.predict_rgb(image_path)
        Image.fromarray(pred_rgb).save(output_path)
        print(f"Маска сохранена: {output_path}")


# ==================== ФУНКЦИЯ ДЛЯ ПАПКИ ====================

def predict_folder(model, input_folder, output_folder):
    """
    Предсказание для всех изображений в папке.
    
    Args:
        model: Загруженная модель DeepGlobeModel
        input_folder: Папка с входными изображениями
        output_folder: Папка для сохранения результатов
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Поддерживаемые форматы
    extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    files = [f for f in os.listdir(input_folder) 
             if os.path.splitext(f)[1].lower() in extensions]
    
    print(f"Найдено {len(files)} изображений в {input_folder}")
    
    for i, filename in enumerate(files):
        print(f"\nОбработка {i+1}/{len(files)}: {filename}")
        
        try:
            image_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            
            # Предсказание и визуализация
            pred, areas = model.visualize_prediction(
                image_path, 
                save_path=os.path.join(output_folder, f"{base_name}_result.png"),
                show=False
            )
            
            # Сохранение только маски
            mask_path = os.path.join(output_folder, f"{base_name}_mask.png")
            pred_rgb = mask_to_rgb(pred)
            Image.fromarray(pred_rgb).save(mask_path)
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")
    
    print(f"\n✅ Готово! Результаты в: {output_folder}")


# ==================== MAIN ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepGlobe Land Cover Inference')
    parser.add_argument('--image', type=str, help='Путь к изображению')
    parser.add_argument('--folder', type=str, help='Путь к папке с изображениями')
    parser.add_argument('--model', type=str, required=True, help='Путь к весам модели (.pth)')
    parser.add_argument('--output', type=str, default='./output', help='Папка для результатов')
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER, help='Название энкодера')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE, help='Размер изображения')
    parser.add_argument('--no_show', action='store_true', help='Не показывать графики')
    
    args = parser.parse_args()
    
    # Загрузка модели
    model = DeepGlobeModel(
        model_path=args.model,
        encoder=args.encoder,
        img_size=args.img_size
    )
    
    # Создание выходной папки
    os.makedirs(args.output, exist_ok=True)
    
    if args.image:
        print(f"\nОбработка: {args.image}")
        pred, class_areas = model.visualize_prediction(
            args.image, 
            save_path=os.path.join(args.output, 'result.png'),
            show=not args.no_show
        )
    
    elif args.folder:
        predict_folder(model, args.folder, args.output)
    
    else:
        print("Укажите --image или --folder")
        return
    
    print("\n✅ Готово!")

