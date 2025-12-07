import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import io


st.set_page_config(page_title="Hair Curl Type Classifier", layout="wide")

def apply_custom_css(css_file):
    with open(css_file) as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

apply_custom_css('streamlit_app/style.css')


st.title("HairNet")
st.header("DSAN-6600: Hair Curl Type Classifier")

st.write("Upload a `.jpg` image to find out what type of hair you have.")
st.write("Hair texture can be classified into 1-4 and a-c, according to Andre Walker's hair-typing system.")
st.write("Make sure the image is centered on your head.")

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Get model state dictionary from S3 bucket
# s3_client = boto3.client('s3')
s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket_name = 'med2106-neural-nets-hair-project'
object_key = "FINALMODEL_weights.pth"
try:
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    body = response["Body"].read()
    print("Downloaded .pth file from S3")
    state_dict = torch.load(io.BytesIO(body), map_location=device)
    print("Unpacked the state_dict from .pth file")
    if "classifier.1.fc.weight" in state_dict:
        state_dict["classifier.1.weight"] = state_dict.pop("classifier.1.fc.weight")
        state_dict["classifier.1.bias"] = state_dict.pop("classifier.1.fc.bias")
except Exception as e:
    print(f"Error downloading .pth: {e}")
    raise

# 3. Re-build the model
weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
model = efficientnet_v2_m(weights=weights)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 9)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model Loaded Sucessfully!")

# 3. File upload
uploaded_file = st.file_uploader(
    "Upload a `.jpg` image",
    type=["jpg"],
    accept_multiple_files=False
)


def validate_input(uploaded_file):
        if not uploaded_file:
            return False, "Please provide a image via upload"
        return True, None

def standardize_image(uploaded_file):
    pil_image = Image.open(uploaded_file).convert("RGB")
    processed_image = transforms.Compose([
        transforms.Resize((600, 600)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return processed_image(pil_image)

labels = ['1', '2a', '2b', '2c', '3a', '3b', '3c', '4a', '4b', '4c']

def predict_hair_type(logits):
    probs = torch.sigmoid(logits)
    class_idx = (probs > 0.5).sum(dim=1).item()
    return labels[class_idx]


def classify_image(image_path):
    # load and transform image
    image = standardize_image(image_path).unsqueeze(0).to(device)
    print("image transformed")
    # ensure model is ready to evaluate an image
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        pred_class = predict_hair_type(outputs)
    return pred_class

hair_care_info = {
    "1": (
        "Straight Hair Info"
    ),
    '2a':(
        "2A hair has a subtle, barely-there texture that's straightforward to straighten. People with this texture should be wary of using heavy styling products that can easily "
        "weigh strands down, making hair look limp. If that sounds like you, Matais-Bernard recommends opting for lightweight products that still offer 'a lot of hold.'To get the job done, "
        "he likes the Davines Curl Moisturizing Mousse. 'It gives the perfect balance of hold and weightlessness,' says the stylist. Because type 2A waves tend to lack volume at the roots, "
        "Dickey recommends using an airy, water-based mousse, like the Aveda Phomollient Styling Foam, to add a bit of oomph at the base."
    ),
    '2b':(
        "2B girlies have hair that lies flatter at the crown and defined “S” waves beginning from mid-lengths. Their strands are thicker than 2A’ers"
        " and require more elbow grease to get hair pin-straight (heat-protectant, please).To enhance your surfer-babe waves, use a texturizing mist "
        "like the Ouai Wave Spray, enriched with rice protein. Ceremonia’s Guava Beach Waves Hair Texturizing Spray is another great option for "
        "moisturizing and amping your natural waves without the weight. “'The struggle with Type 2 curls is longevity, so I always recommend "
        "diffusing for maximum volume and hold for curls that tend to flatten throughout the day,' says Matais-Bernard. We love the Curlsmith "
        "Defrizzion Travel Hair Dryer & Diffuser because it’s excellent at cutting down on frizz and has a large surface area for tackling all your hair at once."
    ),
    '2c':(
        "2C waves are thicker and more susceptible to frizz, with more definition in the “S”-bends that begin at the root. Between shampoos, 2C’ers can opt for a non-lathering, "
        "sulfate-free co-wash to avoid stripping essential moisture from strands. Dickey also recommends layering a leave-in conditioner under a mousse to lock in your "
        "natural wave pattern while adding hydration. We vouch for the Verb Curl Leave-In Conditioner and the Design Essentials Natural Almond & Avocado Curl Enhancing Mousse."
    ),
    '3a':(
        "3A strands tend to be shiny with broader, looser curls that have a diameter about the size of a piece of sidewalk chalk (TBT). To swiftly style 3A hair, work a dollop "
        "or two of curl cream or mousse (like the Best of Beauty Award-winning SGX NYC Curl Power Nourishing Curl Cream) into your damp hair, 'raking it through with your hands "
        "from roots to ends, and scrunching out the excess water,' explains Matais-Bernard. Doing so will help define the curls' texture and hydrate them in the process. Refrain "
        "from touching your hair after applying the products, or you'll risk sparking a frizz halo. Spritz your hair with a curl refresher to maintain those bouncy coils, "
        "like the Carol's Daughter Hair Milk Nourishing & Conditioning Refresher Spray. This formula is lightweight, defining, and smells like yummy, sweet almonds."
    ),
    '3b':(
        "3B hair is made up of springy ringlets with a circumference similar to that of a Sharpie marker. This texture trends dry, so stay stocked with curl gels formulated "
        "with hydration-locking humectants, like hyaluronic acid, glycerin, (familiar skin-care ingredients that serve similar functions), and aloe vera extract to attract moisture."
        "Give the glycerin-rich Mielle Organics Honey & Ginger Styling Gel or the Curls Goddess Botanical Gel, another hydrating and defining pick. Word to the wise: "
        "'Apply when [your hair is] wet, so you'll get definition without frizz,' urges Dickey."
    ),
    '3c':(
        "Type 3C curls resemble tight corkscrews with diameters comparable to straws or pencils. Strands are densely gathered, giving way to lots of "
        "natural volume. Frizziness is to be expected in 3C hair, so if you're trying to mitigate fluff and flyaways, reach for a sulfate-free, non-drying, "
        "creamy cleanser like the Oyin Handmade Ginger Mint Co-Wash. Dickey also likes layering a mousse (such as the 2020 Best of Beauty-winning Rucker "
        "Roots Texture Styling Mousse) over a styling cream (like the Eden BodyWorks Coconut Shea Curl Defining Creme) when the hair is sopping wet to "
        "allow curls to clump together and form faster. 'Your co-wash reveals your curl pattern, while your styling product captures [it],' Dickey explains."
    ),
    '4a':(
        "People with Type 4A hair have dense, springy, 'S'-pattern coils that are about the same circumference as a crochet needle. If this sounds like you, "
        "look to Yara Shahidi and Megan Thee Stallion's texture here for reference. Wash-and-gos are an easy way to style 4A curls. When doing so, Matais-Bernard "
        "recommends 'keeping your detangling brush and sectioning clips handy' for extra ease. This styling method should be done more frequently to keep this "
        "coily texture soft and pliable. A curl cream with a leave-in moisturizer is a must for adding more moisture to daily wash-and-go styling. A curl cream "
        "like Pattern's Styling Cream can be paired with a leave-in like SheaMoisture's Strengthen & Restore Leave-In Conditioner. This combination will "
        "help define your curls without leaving them hard or crunchy. Use a diffuser to dry and disperse your curls for an extra beautiful body."
    ),
    '4b':(
        "Strands with the 4B pattern are densely packed and can bend in sharp angles like the letter 'Z.' 'I love that [4B hair] can be shaped in many "
        "different ways,' says François. Dickey digs styling creams for 4B hair because they're thicker and conducive for palm-rolling (using your "
        "hands to roll your hair into locs or twists) or shingling (using product to manipulate individual curls with your fingers), two types of "
        "product distribution methods that stretch out coils and clump them for better texture definition and elongation."
    ),
    '4c':(
        "4C hair is similar to 4B, but these tightly coiled curls are more fragile and have a tighter zigzag pattern. This hair type experiences "
        "the most significant amount of shrinkage—about 75 percent or more—compared to the other textures. If you are a 4C, take your style cues "
        "from actress Kiki Layne. 'I love that [this texture] is so versatile,' says François. Since shrinkage and dryness are major concerns for "
        "4C'ers, Matais-Bernard recommends using a hair mask, like Mizani Moroccan Clay Steam Mask to 'help soften and detangle while adding intense hydration to thirsty curls.'"
        "After washing, use a liberal amount of leave-in moisturizer, like the Bread Beauty Hair Cream, to supplement moisture. "
        "Castor oil is also a great hydrator and sealant for this dry texture; we like the Briogeo B. Well Cold-Pressed Castor Oil."
    )
}

if st.button("Classify my Hair Type", key="classify_btn"):

    is_valid, msg = validate_input(uploaded_file)
    if not is_valid:
        st.warning(msg)

    try:
        # image = standardize_image(uploaded_file)

        with st.spinner("Classifying your hair...."):
            hair_type = classify_image(uploaded_file)
            pass
        st.success("Hair Texture found!")

        # Display results
        routine = hair_care_info.get(hair_type, "No routine available.")

        st.subheader(f"Hair Texture: {hair_type}")
        st.write(f"**Hair Care Info:**")
        st.write(f"{routine}")
        st.markdown("---")

        st.write("All hair care recommendations from Allure.com")
        st.write("https://www.allure.com/gallery/curl-hair-type-guide")

    except Exception as e:
        st.error(f"Model failed: {e}")

# st.caption(f"Classifier")