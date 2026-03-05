# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:06:42 2026

@author: Endor
"""
import streamlit as st
import nltk
import re
import urllib.request

# --- 1. Setup & Cloud Dictionary Builder ---
@st.cache_resource
def setup_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')  # <-- The crucial fix for the LookupError!

@st.cache_data
def load_and_build_dictionary():
    """Downloads the CMU data and builds your 121k dictionary in the cloud memory"""
    url = "https://raw.githubusercontent.com/kastnerkyle/diphone_synthesizer/master/cmudict.0.7a_SPHINX_40.align"
    try:
        response = urllib.request.urlopen(url)
        lines = response.read().decode('utf-8').splitlines()
        
        temp_dict = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'): continue
            tokens = line.split()
            if len(tokens) < 2: continue
            raw_word = tokens[0].lower()
            if not raw_word[0].isalpha(): continue
            word = raw_word.split('(')[0]
            graphemes = list(word) 
            phonemes = tokens[1:]
            
            if len(graphemes) == len(phonemes):
                word_alignment = []
                for g, p in zip(graphemes, phonemes):
                    g_clean = g if g != '_' else ''
                    p_clean = p if p != '_' else ''
                    if g_clean or p_clean:
                        word_alignment.append([g_clean, p_clean])
                if word not in temp_dict:
                    temp_dict[word] = word_alignment
        return temp_dict
    except Exception as e:
        st.error(f"Cloud build failed: {e}")
        return {}

setup_nltk()
with st.spinner("Initializing linguistic engine..."):
    aligned_dict = load_and_build_dictionary()

# --- 2. Friendly Phoneme Dictionaries ---
VOWELS = {
    "AA": "AA - (e.g., odd, father)", "AE": "AE - (e.g., at, fast)", "AH": "AH - (e.g., hut, up)",
    "AO": "AO - (e.g., ought, caught)", "AW": "AW - (e.g., cow, out)", "AY": "AY - (e.g., hide, my)",
    "EH": "EH - (e.g., red, bed)", "ER": "ER - (e.g., hurt, bird)", "EY": "EY - (e.g., ate, day)",
    "IH": "IH - (e.g., it, sit)", "IY": "IY - (e.g., eat, see)", "OW": "OW - (e.g., oat, go)",
    "OY": "OY - (e.g., toy, boy)", "UH": "UH - (e.g., hood, look)", "UW": "UW - (e.g., two, blue)"
}

CONSONANTS = {
    "B": "B - (e.g., bat, be)", "CH": "CH - (e.g., cheese, catch)", "D": "D - (e.g., dog, day)", 
    "DH": "DH - (e.g., the, father)", "F": "F - (e.g., fish, fee)", "G": "G - (e.g., green, go)",
    "HH": "HH - (e.g., hat, he)", "JH": "JH - (e.g., jump, judge)", "K": "K - (e.g., key, cat)", 
    "L": "L - (e.g., lamp, lee)", "M": "M - (e.g., man, me)", "N": "N - (e.g., no, knee)",
    "NG": "NG - (e.g., sing, running)", "P": "P - (e.g., pen, pee)", "R": "R - (e.g., run, read)", 
    "S": "S - (e.g., sun, sea)", "SH": "SH - (e.g., shoe, she)", "T": "T - (e.g., top, tea)",
    "TH": "TH - (e.g., think, bath)", "V": "V - (e.g., van, vee)", "W": "W - (e.g., water, we)", 
    "Y": "Y - (e.g., yellow, yes)", "Z": "Z - (e.g., zoo, zebra)", "ZH": "ZH - (e.g., measure, vision)"
}

# --- 3. User Interface ---
st.title("English Phoneme Highlighter")
text_input = st.text_area("Enter your text here:", "")

category = st.radio("Sound Category:", ["Vowels", "Consonants"], horizontal=True)
display_options = list(VOWELS.values()) if category == "Vowels" else list(CONSONANTS.values())
selected_display_text = st.selectbox("Choose the specific sound:", display_options)
target_phoneme = selected_display_text.split(" -")[0]

# --- 4. Text Processing Engine ---
if st.button("Highlight Phonemes"):
    words = nltk.word_tokenize(text_input)
    # Run the grammar cop over the tokenized words
    tagged_words = nltk.pos_tag(words)
    highlighted_output = []

    # Loop through both the word AND its grammar tag
    for word, pos_tag in tagged_words:
        if not word.isalnum():
            highlighted_output.append(word)
            continue
            
        lower_word = word.lower()
        if lower_word in aligned_dict:
            # We use list() to create a copy so we don't mutate the master dictionary
            alignment = list(aligned_dict[lower_word])
            
            # --- NEW: Comprehensive Heteronym Grammar Override ---
            if lower_word == "read":
                if pos_tag in ["VBD", "VBN"]:
                    alignment = [['r', 'R'], ['e', 'EH'], ['a', ''], ['d', 'D']]
                else:
                    alignment = [['r', 'R'], ['e', 'IY'], ['a', ''], ['d', 'D']]
            elif lower_word == "record":
                if pos_tag.startswith("VB"):
                    alignment = [['r', 'R'], ['e', 'IH'], ['c', 'K'], ['o', 'AO'], ['r', 'R'], ['d', 'D']]
                else:
                    alignment = [['r', 'R'], ['e', 'EH'], ['c', 'K'], ['o', 'ER'], ['r', 'R'], ['d', 'D']]
            elif lower_word == "object":
                if pos_tag.startswith("VB"):
                    alignment = [['o', 'AH'], ['b', 'B'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']]
                else:
                    alignment = [['o', 'AA'], ['b', 'B'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']]
            elif lower_word == "tear":
                if pos_tag.startswith("VB"):
                    alignment = [['t', 'T'], ['e', 'EH'], ['a', ''], ['r', 'R']]
                else:
                    alignment = [['t', 'T'], ['e', 'IY'], ['a', ''], ['r', 'R']]
            elif lower_word == "live":
                if pos_tag.startswith("VB"):
                    alignment = [['l', 'L'], ['i', 'IH'], ['v', 'V'], ['e', '']]
                else:
                    alignment = [['l', 'L'], ['i', 'AY'], ['v', 'V'], ['e', '']]
            elif lower_word == "lead":
                if pos_tag.startswith("NN"):
                    alignment = [['l', 'L'], ['e', 'EH'], ['a', ''], ['d', 'D']]
                else:
                    alignment = [['l', 'L'], ['e', 'IY'], ['a', ''], ['d', 'D']]
            elif lower_word == "present":
                if pos_tag.startswith("VB"):
                    alignment = [['p', 'P'], ['r', 'R'], ['e', 'IY'], ['s', 'Z'], ['e', 'EH'], ['n', 'N'], ['t', 'T']]
                else:
                    alignment = [['p', 'P'], ['r', 'R'], ['e', 'EH'], ['s', 'Z'], ['e', 'AH'], ['n', 'N'], ['t', 'T']]
            elif lower_word == "project":
                if pos_tag.startswith("VB"):
                    alignment = [['p', 'P'], ['r', 'R'], ['o', 'AH'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']]
                else:
                    alignment = [['p', 'P'], ['r', 'R'], ['o', 'AA'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']]
            elif lower_word == "wind":
                if pos_tag.startswith("VB"):
                    alignment = [['w', 'W'], ['i', 'AY'], ['n', 'N'], ['d', 'D']]
                else:
                    alignment = [['w', 'W'], ['i', 'IH'], ['n', 'N'], ['d', 'D']]
            elif lower_word == "minute":
                if pos_tag.startswith("JJ"): # Adjective (tiny)
                    alignment = [['m', 'M'], ['i', 'AY'], ['n', 'N'], ['u', 'UW'], ['t', 'T'], ['e', '']]
                else: # Noun (time)
                    alignment = [['m', 'M'], ['i', 'IH'], ['n', 'N'], ['u', 'AH'], ['t', 'T'], ['e', '']]
            # ---------------------------------------------
            
            highlights = [False] * len(alignment)
            
            # 1. Base Matches
            for i, (g, p) in enumerate(alignment):
                if re.sub(r'\d+', '', p) == target_phoneme:
                    highlights[i] = True
                    
            # --- NEW: The 'X' Spillover Fix ---
            for i in range(len(alignment)):
                if alignment[i][0] == 'x':
                    # If we are searching for an 'x' sound (K, S, G, or Z)
                    if target_phoneme in ['K', 'S', 'G', 'Z']:
                        # Check if the next letter accidentally caught the spillover sound
                        if i + 1 < len(alignment) and re.sub(r'\d+', '', alignment[i+1][1]) == target_phoneme:
                            highlights[i] = True     # Highlight the 'x'
                            highlights[i+1] = False  # Remove highlight from the innocent next letter
            # ----------------------------------
            
            # 2. Your Ultimate Multi-Letter Catcher Logic
            tetraph_rules = {"tion": ["SH","AH","N"], "sion": ["SH","ZH","AH","N"], "eigh": ["EY"], "augh": ["AO","F"], "ough": ["OW","AW","UW","AO","F","AH"]}
            trigraph_rules = {"igh": ["AY"], "tch": ["CH"], "dge": ["JH"], "eau": ["OW","UW"], "ous": ["AH","S"], "que": ["K"]}
            pair_rules = {"sh":["SH"],"ch":["CH","K","SH"],"th":["TH","DH"],"ph":["F"],"wh":["W","HH"],"ng":["NG"],"gh":["F","G"],"ck":["K"],"kn":["N"],"wr":["R"],"mb":["M"],"gn":["N"],"rh":["R"],"ti":["SH"],"ci":["SH"],"si":["SH","ZH"],"ce":["SH"],"tu":["CH"],"su":["SH","ZH"],"ea":["IY","EH","EY"],"ee":["IY"],"oa":["OW"],"oo":["UW","UH"],"ou":["AW","AH","UW","OW"],"ow":["AW","OW"],"ai":["EY","EH"],"ay":["EY"],"ei":["EY","IY"],"ey":["EY","IY"],"au":["AO"],"aw":["AO"],"ew":["UW","Y"],"oe":["OW","UW"],"ie":["IY","AY"],"ui":["UW","IH"],"ue":["UW"]}

            # 4-letter combos
            for i in range(len(alignment) - 3):
                quad = "".join([a[0] for a in alignment[i:i+4]])
                if quad in tetraph_rules and target_phoneme in tetraph_rules[quad]:
                    if any(highlights[i:i+4]): highlights[i:i+4] = [True, True, True, True]

            # 3-letter combos
            for i in range(len(alignment) - 2):
                triple = "".join([a[0] for a in alignment[i:i+3]])
                if triple in trigraph_rules and target_phoneme in trigraph_rules[triple]:
                    if any(highlights[i:i+3]): highlights[i:i+3] = [True, True, True]

            # 2-letter combos & Doubles
            for i in range(len(alignment) - 1):
                pair = "".join([a[0] for a in alignment[i:i+2]])
                is_double = (alignment[i][0] == alignment[i+1][0] and alignment[i][0].isalpha())
                if (pair in pair_rules and target_phoneme in pair_rules[pair]) or is_double:
                    if any(highlights[i:i+2]): highlights[i:i+2] = [True, True]

            # 3. Final Render
            word_html = "".join([f"<span style='background-color: #FFFF00; font-weight: bold; color: black; padding: 0 2px; border-radius: 3px;'>{g}</span>" if highlights[i] else g for i, (g, p) in enumerate(alignment)])
            highlighted_output.append(word_html)
        else:
            highlighted_output.append(word)

    final_html = re.sub(r' ([.,!?\'])', r'\1', " ".join(highlighted_output))
    st.markdown("### Result:")

    st.markdown(f"<div style='font-size: 24px; line-height: 1.5;'>{final_html}</div>", unsafe_allow_html=True)
