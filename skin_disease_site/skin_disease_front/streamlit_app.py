import streamlit as st
from PIL import Image
import numpy as np
import base64
import requests
import json
import sys



#Setting Title of App
st.title("Skin Disease Detection")

IMAGE_SHAPE = (128, 128)  # must be the same as the input of the trained model

#api_url = sys.argv[1]+'/predict'
api_url = 'http://localhost:8000' + '/predict'

print(api_url)
logo = Image.open("logo.png")
st.image(
    logo,
    width=200,
)

st.set_option('deprecation.showfileUploaderEncoding', False)

#Uploading the image
uploaded_file = st.file_uploader("Choose an image...",
                                 type=['png', 'jpeg', 'jpg'])
submit = st.button('Predict')

if uploaded_file:

    #st.image(uploaded_file)
    st.image(uploaded_file,
             channels="RGB",
             output_format='PNG',
             use_column_width=True)

    test_pic = Image.open(uploaded_file).convert('RGB')
    #turn to array
    test_pic = test_pic.resize(
        IMAGE_SHAPE)  # this must be the input shape of the network

    test_pic_array = np.array(test_pic)
    #switch to U-Int 8
    test_pic_array = test_pic_array.astype('uint8')
    #memorizing shape
    height, width, channel = test_pic_array.shape
    #reshape
    test_pic_array = test_pic_array.reshape(height * width * channel)
    # encoding to b64
    b64bytes = base64.b64encode(test_pic_array)
    #decoding to utf8 and turning to  string
    b64str = b64bytes.decode('utf8').replace("'", '"')
    image_dict = {
        'image': b64str,
        'height': height,
        'width': width,
        'channel': channel
    }
    headers = {'Content_Type': 'application/json'}

    newjson = json.dumps(image_dict)
    response = requests.post(api_url, newjson, headers=headers).json()

    response

    st.write(str(response))
