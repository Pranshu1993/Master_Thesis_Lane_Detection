import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn   
import tempfile
import json
import pandas as pd
import gdown

# Load your pre-trained models here (replace with your actual models)

class ResNet18_LaneDetection(nn.Module):
    def __init__(self):
        super(ResNet18_LaneDetection, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 224 * 224)
        

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, 224, 224)

        return x
    
class ResNet34_LaneDetection(nn.Module):
    def __init__(self):
        super(ResNet34_LaneDetection, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 224 * 224)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 1, 224, 224)

        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc1 = self.encoder(x)
        enc2 = self.middle(enc1)
        dec = self.decoder(enc2)
        return dec

class DeepLabModel(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabModel, self).__init__()

        # Load the pre-trained DeepLabV3 model with ResNet-50 backbone
        self.model = deeplabv3_resnet50(pretrained=True, progress=True)
        
        # Modify the classifier's last layer for the desired number of classes
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        # Forward pass through the DeepLabV3 model
        return self.model(x)
    
# Load & set Model
def load_and_set_model(model_path, model_class, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def download_from_gdrive(file_id, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Model Paths
# resnet_18_path = 'D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\models\\resnet_18_model.pth'
# resnet_34_path = 'D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\models\\resnet_34_model.pth'
# UNET_path      = 'D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\models\\U_Net_model.pth'
# DeeplabV3_path = 'D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\models\\DeepLabV3_model.pth'

# file ID
file_ids = ['1zK2xKBE0NlsqJzhjecNq9xreKy7141PE','1adNRZn7tCxgz4tHGnqJs-D8AqqSt1kjm','1QGK2CzVrbr0lnCAKFoCAgwcT3GDnFqFO','1ueO9PDwzt2gu1Rks2M_Zv4yeKWgElviA']
# resnet_18_File_ID = '1zK2xKBE0NlsqJzhjecNq9xreKy7141PE'
# resnet_34_File_ID = '1adNRZn7tCxgz4tHGnqJs-D8AqqSt1kjm'
# UNet_File_ID      = '1QGK2CzVrbr0lnCAKFoCAgwcT3GDnFqFO'
# DeeplabV3_File_ID = '1ueO9PDwzt2gu1Rks2M_Zv4yeKWgElviA'

output_paths = [
    '/tmp/models/resnet_18_model.pth',
    '/tmp/models/resnet_34_model.pth',
    '/tmp/models/U_Net_model.pth',
    '/tmp/models/DeepLabV3_model.pth'
]

model_classes = [ResNet18_LaneDetection, ResNet34_LaneDetection, UNet, DeepLabModel]

# Additional arguments for each model (e.g., num_classes for DeepLabModel)
model_args = [[], [], [], [1]]

# Download, load, and set each model
for file_id, output_path, model_class, args in zip(file_ids, output_paths, model_classes, model_args):      
    # Download model from Google Drive
    download_from_gdrive(file_id, output_path)
    
    # Load and set the model using the provided function
    if model_class == DeepLabModel:
        model = load_and_set_model(output_path, model_class, num_classes=args[0])
    else:
        model = load_and_set_model(output_path, model_class)
    
    # For instance, setting them to variables as per your previous code:
    if model_class == ResNet18_LaneDetection:
        resnet_18_model = model
    elif model_class == ResNet34_LaneDetection:
        resnet_34_model = model
    elif model_class == UNet:
        unet_model = model
    elif model_class == DeepLabModel:
        deeplab_model = model


# Move the models to the same device as the input data
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

resnet_18_model.to(device)
resnet_34_model.to(device)
unet_model.to(device)
deeplab_model.to(device)


# Load and set models to evaluation mode
# resnet_18_model = load_and_set_model(resnet_18_path, ResNet18_LaneDetection)
# resnet_34_model = load_and_set_model(resnet_34_path, ResNet34_LaneDetection)
# unet_model = load_and_set_model(UNET_path, UNet)
# deeplab_model = load_and_set_model(DeeplabV3_path, DeepLabModel, num_classes=1)


# Define a function for preprocessing the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

@st.cache_data
def download_file_from_gdrive(url, output_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_id = url.split("/file/d/")[1].split("/view")[0]
    gdown_url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(gdown_url, output_path, quiet=False)

#Load JSON file
# def load_json_file(file_path):
#     """Loads a JSON file and returns a Pandas DataFrame."""
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     df = pd.DataFrame(data)
#     return df

@st.cache_data
def load_json_file(file_url):
    """Loads a JSON file from Google Drive and returns a Pandas DataFrame."""
    # Define the temporary path to save the downloaded file.
    temp_path = '/tmp/test_Json.json'
    
    # Download the file
    download_file_from_gdrive(file_url, temp_path)
    
    # Load and return the DataFrame
    with open(temp_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df



def get_matching_row(df, input_image_name):
    """
    Function to match the input image name to its corresponding entry in the dataframe.
    
    Args:
    - df (pd.DataFrame): The dataframe containing 'raw_file' column from test_label_JSON.
    - input_image_name (str): The name of the input image.
    
    Returns:
    - pd.Series: The matched row from the dataframe.
    """
    
    # Extract parts from the input image name
    input_image_parts = input_image_name.rsplit('.', 1)[0].split('_')  # Removing the extension and splitting by "_"
    
    # Construct the expected pattern for matching in the dataframe
    expected_pattern = f"clips/{input_image_parts[0]}/{input_image_parts[1]}_{input_image_parts[2]}/"
    
    # Searching for a match in the dataframe
    matching_rows = df[df['raw_file'].str.contains(expected_pattern)]
    
    if len(matching_rows) > 0:
        return matching_rows.iloc[0]
    else:
        raise ValueError(f"No matching entry found for {input_image_name} in the dataframe!")




def generate_lane_mask(row):
    
    mask = np.zeros((720, 1280, 3))
    h_samples = row['h_samples']
    lanes = row['lanes']
    raw_file = row['raw_file']

    # create mask: lane: 1, non-lane: 0
    for lane in lanes:    
        
        h_samples_updated = [y for x, y in zip(lane, h_samples) if x != -2]
        lane = [x for x in lane if x != -2]
        lane_points = np.array(list(zip(lane, h_samples_updated)))
        # add lane markings to the mask we created
        cv2.polylines(mask, [lane_points], False, (255, 255, 255), thickness=15)
        
        # write the lane mask to the desired directory

        # path = 'D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\TuSimple_Dataset\\TUSimple\\test_set\\tusimple_preprocessed_test\\lane-masks'  # Shortened for readability
        # path = 'https://drive.google.com/drive/folders/1uafeVGyHh5d2AKG5h4ms2Q276LTvsS_7?usp=drive_link'

        # Set the path to Streamlit's temporary directory
        path = '/tmp/lane_masks'

        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Name each mask according to its image's name
        tmp = raw_file[:-7].split('/')[-2:]
        mask_fname = f'{tmp[0]}_{tmp[1]}.jpg'
        new_file = os.path.join(path, mask_fname)
        cv2.imwrite(new_file, mask)
        
    return mask_fname, path 


def compute_metrics(true_mask, pred_mask):
    true_mask_bool = true_mask > 0
    pred_mask_bool = pred_mask > 0

    intersection = np.logical_and(true_mask_bool, pred_mask_bool)
    union = np.logical_or(true_mask_bool, pred_mask_bool)

    iou = np.sum(intersection) / np.sum(union)

    tp = np.sum(intersection)  
    fp = np.sum(np.logical_and(np.logical_not(true_mask_bool), pred_mask_bool))  
    fn = np.sum(np.logical_and(true_mask_bool, np.logical_not(pred_mask_bool)))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return iou, precision, recall, f1_score

def get_arrow(current_value, prev_value):
    if prev_value is None:
        return "-"
    elif current_value > prev_value:
        return '<span style="color:green; font-size:34px;">↑</span>'
    elif current_value < prev_value:
        return '<span style="color:red; font-size:34px;">↓</span>'
    else:
        return "-"
    


# df_test_json = load_json_file('D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\TuSimple_Dataset\\TUSimple\\test_label.json')
df_test_json = load_json_file('https://drive.google.com/file/d/1o_hhB9y96pAnGXclUlmO7CiZ0LKHF8v-/view?usp=sharing')




st.sidebar.header('Navigation')
st.title("Lane Detection - Guiding Your Journey")

user_role = st.sidebar.selectbox("Select User Role", ['Home',"Normal User", "Data Scientist"])

if user_role == 'Home':
    st.markdown("""
        Welcome to the Lane Detection app, where technology meets the road. Leveraging cutting-edge machine learning models, this tool helps in identifying and marking lanes on the road to aid autonomous vehicles and enhance road safety. Feel free to explore and witness the convergence of artificial intelligence and road safety.
        """)

    # Adding a banner image (replace 'path/to/banner.jpg' with the actual path to your banner image)
    # file_path = 'D:\\Pranshu\\MS_Colleges\\SRH_Hiedelberg\\Semester_4\\Master_Thesis\\Topic\\Prof_Swati_Topics\\Lane_Detection\\TF_LD_env\\UI\\Banner.jpg'
    Banner_file_path = 'https://drive.google.com/file/d/1IFclSTzuIn91BVftQMOPxeQPC1i7CEqh/view?usp=sharing'

    # Define the temporary output path on Streamlit Cloud
    Banner_path = '/tmp/Banner.jpg'

    # Use the function to download the file
    download_file_from_gdrive(Banner_file_path  , Banner_path)

    try:
        img = Image.open(Banner_path)
        st.image(img, use_column_width=True)
    except Exception as e:
        st.write(f"Could not open image: {e}")

    st.markdown("""
        ### Features:
        - **Lane Detection**: Upload a road image and get the lanes detected instantly.
        - **Model Insights**: Deep dive into how different models perceive lanes and their accuracies.
        - **Real-time Analysis**: Analyze road footage in real-time and see how the models perform.

        ### Get Started:
        To get started, select a user role from the sidebar and navigate through the app functionalities available for that role. Feel free to explore !

        Thank you for visiting!
        """)


if user_role == "Normal User":
    st.title("Normal User View")

    # Model Selection Drop Down
    model_option = st.sidebar.selectbox("Select a Model", ["ResNet-18", "ResNet-34", "U-net", "DeepLabv3"])

    # Image uploader at the center top
    st.write("Image Uploader Top")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="uploader")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        img = Image.open(uploaded_file)

        # Placeholder for processed image display
        st.write("Processed Image:")

        # Select the model based on the dropdown selection (replace with your actual models)
        if model_option == "ResNet-18":
            selected_model = resnet_18_model
        elif model_option == "ResNet-34":
            selected_model = resnet_34_model
        elif model_option == "U-net":
            selected_model = unet_model
        elif model_option == "DeepLabv3":
            selected_model = deeplab_model

        # Preprocess the uploaded image
        input_tensor = preprocess_image(img).to(device)

        # Perform inference using the loaded model
        with torch.no_grad():
            if selected_model == deeplab_model:
                tensor_output = selected_model(input_tensor)
                tensor_output = tensor_output['out']
            else:
                tensor_output = selected_model(input_tensor)

        # Post-process the model's output and display results in Streamlit
        pred_mask = (torch.sigmoid(tensor_output) > 0.4).cpu().numpy().squeeze().astype(np.uint8)

        # Resize the boolean mask to match image dimensions (1280x720)
        resized_mask = cv2.resize(pred_mask, (1280, 720))

        # Create a copy of the input image to draw the lanes on
        output_img = np.array(img.copy())

        # Find the contours of the detected lane mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

        # Draw the detected lanes as thin green lines on the output image
        cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)

        # Display the output image
        st.image(output_img, caption="Lane Detection Output", use_column_width=True)

if user_role == "Data Scientist":
    st.title("Data Scientist View")

    # Model Selection Drop Down
    model_option = st.sidebar.selectbox("Select a Model", ["ResNet-18", "ResNet-34", "U-net", "DeepLabv3"])

    # Image uploader at the center top
    st.write("Image Uploader Top")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="uploader")

    # Display the uploaded image
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        img = Image.open(uploaded_file)

        # Save the image to a temporary location and retrieve the path
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        img.save(temp_path)

        # Extract the file name from the path
        input_image_name = os.path.basename(temp_path)

        # Use the function to get the matching row
        row = get_matching_row(df_test_json, input_image_name)
        mask_fname, path = generate_lane_mask(row)

        print(mask_fname)

        ground_truth_mask_path = os.path.join(path, mask_fname)
        ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)

        ground_truth_mask = (ground_truth_mask > 127).astype(np.uint8)  # Convert to 0 or 1

        # Placeholder for processed image display
        st.write("Processed Image:")

        # Select the model based on the dropdown selection (replace with your actual models)
        if model_option == "ResNet-18":
            selected_model = resnet_18_model
        elif model_option == "ResNet-34":
            selected_model = resnet_34_model
        elif model_option == "U-net":
            selected_model = unet_model
        elif model_option == "DeepLabv3":
            selected_model = deeplab_model
        
        # Preprocess the uploaded image
        input_tensor = preprocess_image(img).to(device)

        # Perform inference using the loaded model
        with torch.no_grad():
            if selected_model == deeplab_model:
                tensor_output = selected_model(input_tensor)
                tensor_output = tensor_output['out']
            else:
                tensor_output = selected_model(input_tensor)

        # Post-process the model's output and display results in Streamlit
        pred_mask = (torch.sigmoid(tensor_output) > 0.5).cpu().numpy().squeeze().astype(np.uint8)

        # Resize the boolean mask to match image dimensions (720x720)
        resized_mask = cv2.resize(pred_mask, (1280, 720))

        # Create a copy of the input image to draw the lanes on
        output_img = np.array(img.copy())

        # Find the contours of the detected lane mask
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

        # Draw the detected lanes as thin green lines on the output image
        cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)

        iou, precision, recall, f1_score = compute_metrics(ground_truth_mask, resized_mask)

        # Creating columns to display the output image and metrics side by side
        col1, col2, col3 = st.columns([2.5,1,0.5])
        
        # Displaying the output image in the first column
        col1.image(output_img, caption="Output Image.", use_column_width=True)
        
        # Displaying the metrics in the second column

        col2.metric(label="IoU", value=f"{iou*100:.2f}%")
        col2.metric(label="Precision", value=f"{precision*100:.2f}%")
        col2.metric(label="Recall", value=f"{recall*100:.2f}%")
        col2.metric(label="F1-Score", value=f"{f1_score*100:.2f}%")

        # Collecting metrics in a dictionary for easier comparison
        metrics = {
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
            }
        
        for metric, value in metrics.items():
            # Initialize session state for first-time use
            if f'prev_{metric}' not in st.session_state:
                st.session_state[f'prev_{metric}'] = value

            # Get the arrow and its color
            arrow_html = get_arrow(value, st.session_state[f'prev_{metric}'])

            # Add two line spaces
            col3.write("")
            
            # Display the metric with the arrow using markdown
            col3.markdown(f"{arrow_html}", unsafe_allow_html=True)

            # Update the previous metric value for the next iteration
            st.session_state[f'prev_{metric}'] = value