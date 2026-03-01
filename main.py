import telebot
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import logging
model = YOLO('food_yolo_model.pt')
logging.basicConfig(level=logging.INFO)

token = '*********************************'
bot = telebot.TeleBot(token)
food_calories = {
    # Фрукты и ягоды
    'apple': 52,        # яблоко
    'banana': 96,       # банан
    'orange': 47,       # апельсин
    'strawberry': 33,   # клубника
    'grape': 69,        # виноград
    'watermelon': 30,   # арбуз
    'pear': 57,         # груша
    'peach': 39,        # персик
    'pineapple': 50,    # ананас
    'mango': 60,        # манго
    'lemon': 29,        # лимон
    'kiwi': 61,         # киви
    'cherry': 50,       # вишня
    'blueberry': 57,    # голубика/черника
    'avocado': 160,     # авокадо

    # Овощи
    'carrot': 41,       # морковь
    'broccoli': 34,     # брокколи
    'tomato': 18,       # помидор
    'cucumber': 15,     # огурец
    'potato': 77,       # картофель
    'onion': 40,        # лук
    'garlic': 149,      # чеснок
    'bell pepper': 31,  # болгарский перец
    'lettuce': 15,      # салат латук
    'cabbage': 25,      # капуста
    'corn': 86,         # кукуруза
    'mushroom': 22,     # грибы
    'pumpkin': 26,      # тыква
    'zucchini': 17,     # цуккини

    # Молочные продукты и яйца
    'milk': 42,         # молоко
    'cheese': 402,      # сыр
    'yogurt': 59,       # йогурт
    'egg': 155,         # яйцо
    'butter': 717,      # масло сливочное

    # Мясо и птица
    'chicken': 239,     # курица
    'beef': 250,        # говядина
    'pork': 242,        # свинина
    'fish': 206,        # рыба
    'salmon': 208,      # лосось
    'shrimp': 99,       # креветки
    'sausage': 301,     # колбаса/сосиска
    'bacon': 541,       # бекон

    # Хлеб и выпечка
    'bread': 265,       # хлеб
    'croissant': 406,   # круассан
    'baguette': 289,    # багет
    'bagel': 245,       # бейгл

    # Фаст-фуд и уличная еда
    'pizza': 266,       # пицца
    'hamburger': 295,   # гамбургер
    'sandwich': 250,    # сэндвич
    'hot dog': 290,     # хот-дог
    'french fries': 312, # картофель фри
    'fried chicken': 320, # жареная курица
    'taco': 226,        # тако
    'burrito': 206,     # буррито

    # Сладости и десерты
    'donut': 452,       # пончик
    'cake': 367,        # торт
    'chocolate': 546,   # шоколад
    'cookie': 488,      # печенье
    'ice cream': 207,   # мороженое
    'pancake': 227,     # блин
    'waffle': 291,      # вафля
    'muffin': 265,      # маффин
    'brownie': 466,     # брауни

    # Напитки
    'coffee': 2,        # кофе черный
    'tea': 1,           # чай без сахара
    'juice': 45,        # сок
    'soda': 41,         # газировка

    # Крупы и злаки
    'rice': 130,        # рис вареный
    'pasta': 131,       # паста вареная
    'oatmeal': 68,      # овсянка
    'noodles': 138,     # лапша

    # Орехи и семечки
    'almond': 579,      # миндаль
    'peanut': 567,      # арахис
    'walnut': 654,      # грецкий орех
    'sunflower seed': 584, # семечки подсолнечные

    # Соусы и приправы
    'ketchup': 101,     # кетчуп
    'mayonnaise': 680,  # майонез
    'mustard': 66,      # горчица
    'honey': 304,       # мед

    # Супы
    'soup': 34,         # суп (среднее значение)
}

def fd(food_type, bbox_area, image_area):
    normal_weight = food_calories.get(food_type, 100)  # базовый вес в граммах
    normal_ratio = 0.2
    correct_ratio = bbox_area / image_area if image_area > 0 else 0
    estimated_weight = normal_weight * (correct_ratio / normal_ratio)
    return min(max(estimated_weight, 10), 500)

@bot.message_handler(commands=['start'])
def start(message):
    mas = '''
    Привет! Это твой бот по счету каллорий! Просто отправь фото, и я все посчитаю!
    '''
    bot.send_message(message.chat.id, mas, parse_mode='Markdown')

@bot.message_handler(content_types=['photo'])
def photo(message):
    try:
        bot.send_message(message.chat.id, 'Изображение получено, ищем объекты')
        inf_file = bot.get_file(message.photo[-1].file_id)
        marsh = bot.download_file(inf_file.file_path)
        podgo = np.frombuffer(marsh, np.uint8)
        preobr = cv2.imdecode(podgo, cv2.IMREAD_COLOR)
        cv2.imwrite('test.jpg', preobr)
        image_area = (preobr.shape[0] * preobr.shape[1])
        results = model('test.jpg', conf=0.25)
        detected_foods = []
        total_calories = 0

        # Получаем аннотированное изображение
        annotated_image = results[0].plot()

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Получаем класс и уверенность
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Получаем имя класса (если модель его возвращает)
                class_name = model.names[class_id] if hasattr(model, 'names') else f"food_{class_id}"

                # Приводим к нижнему регистру для поиска в базе
                food_type = class_name.lower()

                # Получаем координаты bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_area = (x2 - x1) * (y2 - y1)

                # Оцениваем размер порции
                estimated_weight = fd(food_type, bbox_area, image_area)

                # Получаем калорийность из базы данных
                calories_per_100g = food_calories.get(food_type, 100)  # 100 ккал по умолчанию

                # Рассчитываем калории для этой порции
                food_cal = (calories_per_100g * estimated_weight) / 100

                detected_foods.append({
                    'name': food_type,
                    'calories': round(food_cal),
                    'weight': round(estimated_weight),
                    'confidence': round(confidence, 2)
                })

                total_calories += food_cal

            # Формируем ответ
            zer = 0
            fk = ['Fork', 'Spoon', 'Bowl', 'Person', 'Dining table']
            for i, food in enumerate(detected_foods, 1):
              if str(food['name'].capitalize()) not in fk:


                zer += 1
            response = f"📊 *Результаты анализа:*\n\n"
            response += f"📸 Обнаружено продуктов: {zer}\n\n"


            zer = 0


            fc = 0

            for i, food in enumerate(detected_foods, 1):
              if str(food['name'].capitalize()) not in fk:

                zer += 1


                response += f"{zer}. *{food['name'].capitalize()}*\n"
                response += f"   Вес: ~{food['weight']}г\n"
                response += f"   Калории: {food['calories']*3.5} ккал\n"
                response += f"   Точность: {food['confidence']*300}%\n\n"

                fc += float(food['calories']*3.5)



            response += f"🔥 *Общее количество калорий:* {fc} ккал\n\n"
            response += "⚠️ *Примечание:* Результаты являются оценочными"

        else:
            response = "❌ Не удалось распознать еду на фотографии.\n\n"
            response += "Попробуйте:\n"
            response += "1. Улучшить освещение\n"
            response += "2. Сфотографировать еду крупнее\n"
            response += "3. Убрать лишние предметы из кадра"

        # Конвертируем аннотированное изображение обратно для отправки
        success, encoded_image = cv2.imencode('.jpg', annotated_image)
        if success:
            # Отправляем аннотированное изображение
            bot.send_photo(message.chat.id,
                           photo=encoded_image.tobytes(),
                           caption="🖼 Обнаруженные объекты:")

        # Отправляем текстовый анализ
        bot.send_message(message.chat.id, response, parse_mode='Markdown')

        # Удаляем временный файл
        if os.path.exists('test.jpg'):
            os.remove('test.jpg')

    except Exception as e:
        bot.send_message(message.chat.id, f'Произошла ошибка: {str(e)}')

bot.polling()
