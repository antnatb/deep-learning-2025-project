import os
import ast
import re
from time import sleep
from dotenv import load_dotenv
import json
import google.generativeai as genai


CLASS_NAMES = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]


load_dotenv()

# Grab API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Pick a Gemini model (light and free-friendly: gemini-1.5-flash)
model = genai.GenerativeModel("gemini-2.5-flash")
aliases_dict = {}

# Ask a question
prompt = """You are a concise, factual assistant that returns only lists of aliases (no explanations).

Instructions:
1) Given the canonical Flower/Plant name after the colon, return ONLY its common aliases or other names typically used to identify that specific flower.
2) Output must be a valid Python list literal (i.e. start with '[' and end with ']'). Example format exactly:
   Flower/Plant name: Chrysanthemums
   List of alternative names: ['Mums', 'Chrysanths']
3) Do NOT include any text besides the Python list for the requested flower (no labels, no punctuation outside the list, no trailing whitespace, no comments).
4) If there are no known aliases, return an empty list: []
5) Preserve capitalization as commonly used for each alias; include multi-word aliases as single strings.
6) If an alias is identical to the input name, do NOT repeat it.
7) If you are unsure about a name, prefer returning fewer aliases rather than guessing. Do not hallucinate.

Few-shot examples (must be followed exactly):
Flower/Plant name: Chrysanthemums
List of alternative names: ['Mums', 'Chrysanths']

Flower/Plant name: Bellis perennis
List of alternative names: ['English Daisy', 'Lawn Daisy', 'Common Daisy']

Now complete for the requested flower:
Flower/Plant name: {} 
List of alternative names:"""
for flower in CLASS_NAMES:
    formatted_prompt = prompt.format(flower)
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = model.generate_content(formatted_prompt)
            print(response.text)
            aliases = response.text.strip()
            
            # Use regex to extract only the list content between brackets
            list_pattern = r'\[.*?\]'
            match = re.search(list_pattern, aliases)
            
            if match:
                list_string = match.group(0)
                aliases_list = ast.literal_eval(list_string)
                aliases_dict[flower] = aliases_list
                break  # Success, exit retry loop
            else:
                raise ValueError("No list pattern found in response")
        except ValueError as ve:
            continue  # Retry on parsing errors
        except Exception as e:
            print(e)
            retry_count += 1
            print(f"Error processing {flower} (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                print("Waiting 60 seconds before retry...")
                sleep(60)
            else:
                print(f"Failed to process {flower} after {max_retries} attempts")

# Parse output
for flower, aliases in aliases_dict.items():
    print(f"Flower/Plant name: {flower}. List of alternative names: {aliases}"
)
# save dict as json
with open("flower_aliases.json", "w") as f:
    json.dump(aliases_dict, f)