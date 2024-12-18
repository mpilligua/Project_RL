from PIL import Image, ImageDraw, ImageFont
import numpy as np
import seaborn

q_values = np.array([[-0.0035,  0.0563, -0.0527, -0.0013, -0.0228, -0.0561, -0.0148, -0.0304,
          0.0124,  0.0121,  0.0397,  0.0292,  0.0451,  0.0105,  0.0085,  0.0019,
         -0.0198,  0.0238]])

q_values_names = {0:"Noop",
                    1:"Fire",
                    2:"Up",
                    3:"Right",
                    4:"Left",
                    5:"Down",
                    6:"UpRight",
                    7:"UpLeft",
                    8:"DownRight",
                    9:"DownLeft",
                    10:"UpFire",
                    11:"RightFire",
                    12:"LeftFire",
                    13:"DownFire",
                    14:"UpRightFire",
                    15:"UpLeftFire",
                    16:"DownRightFire",
                    17:"DownLeftFire"}

text_height = 12
imgs_width = 370
imgs_height = 220 
combined_height = imgs_height + text_height
q_values_width = (combined_height-text_height) // q_values.shape[1]
total_width = imgs_width + q_values_width

# Create a blank canvas with extra space for text
combined_image = Image.new("RGB", (total_width, combined_height), color=(255, 255, 255))
font = ImageFont.load_default() 
draw = ImageDraw.Draw(combined_image)

# use seaborn to set a palete that we will use to color the boxes based on the q-values
colors = seaborn.color_palette("coolwarm", as_cmap=True)

max_q_value = 1
min_q_value = -1

for val in range(q_values.shape[1]):
    normalized_q_values = (q_values[0, val] - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
    color = tuple(int(255 * c) for c in colors(normalized_q_values))
    box = Image.new("RGB", (q_values_width, q_values_width), color=color)
    combined_image.paste(box, (imgs_width, text_height+(q_values_width*val)))
    # add text in the middle with the q-value
    q_value_name = q_values_names[val]
    draw.text((imgs_width + q_values_width//2, text_height + (q_values_width*val) + q_values_width//2), q_value_name, font=font, fill=(0, 0, 0))


# save the image
combined_image.save("combined_image.png")
