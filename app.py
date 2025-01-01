GRADIO_TITLE = "Langues Guesser based on Name"
GRADIO_DESCRIPTION = '''
This is a self-learning project which replicates the [pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) with modifications.
Kindly see [my GitHub: you may speak](https://github.com/LunaticMaestro/-NLP-_you_may_speak) readme to checkout the modifications.

Model Trained for names on following languages: ['Korean 🇰🇷', 'Portuguese 🇵🇹', 'Dutch 🇳🇱', 'Italian 🇮🇹', 'German 🇩🇪', 'Scottish 🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f', 'Vietnamese 🇻🇳', 'French 🇫🇷', 'English 🇬🇧', 'Arabic 🇲🇦', 'Irish 🇮🇪', 'Chinese 🇨🇳', 'Japanese 🇯🇵', 'Russian 🇷🇺', 'Polish 🇵🇱', 'Czech 🇨🇿', 'Spanish 🇪🇸', 'Greek 🇬🇷']

'''

import gradio as gr
from z_modelops import NameToLanguages
from z_inference import setup_inference, infer_lang

model, labels = setup_inference()

def get_langauge(name):
    langugages = infer_lang(name, model, labels)

    language_flags = {
        "Korean": "\U0001F1F0\U0001F1F7",    # South Korea 
        "Portuguese": "\U0001F1F5\U0001F1F9", # Portugal
        "Dutch": "\U0001F1F3\U0001F1F1",      # Netherlands
        "Italian": "\U0001F1EE\U0001F1F9",    # Italy
        "German": "\U0001F1E9\U0001F1EA",     # Germany
        "Scottish": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F", # Scotland (flag sequence)
        "Vietnamese": "\U0001F1FB\U0001F1F3",  # Vietnam
        "French": "\U0001F1EB\U0001F1F7",      # France
        "English": "\U0001F1EC\U0001F1E7", # England (flag sequence)
        "Arabic": "\U0001F1F2\U0001F1E6",      # UAE (commonly associated with Arabic)
        "Irish": "\U0001F1EE\U0001F1EA",       # Ireland
        "Chinese": "\U0001F1E8\U0001F1F3",     # China
        "Japanese": "\U0001F1EF\U0001F1F5",    # Japan
        "Russian": "\U0001F1F7\U0001F1FA",     # Russia
        "Polish": "\U0001F1F5\U0001F1F1",      # Poland
        "Czech": "\U0001F1E8\U0001F1FF",       # Czech Republic
        "Spanish": "\U0001F1EA\U0001F1F8",     # Spain
        "Greek": "\U0001F1EC\U0001F1F7"         # Greece
    }

    return '\n'.join([lang + " " + language_flags[lang] for lang in langugages])


input_textbox = gr.Textbox(label="Your Name", placeholder="Naifeh", max_lines=1)


demo = gr.Interface(
    fn=get_langauge, 
    inputs=input_textbox ,
    outputs=gr.Label(label="You may speak"),
    title=GRADIO_TITLE,
    description=GRADIO_DESCRIPTION
)
demo.launch(debug=True)   