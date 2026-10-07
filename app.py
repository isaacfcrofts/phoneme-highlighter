# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:06:42 2026

@author: Endor
"""
# --- 1. Setup & ML Model ---
import streamlit as st
import nltk
import re
from g2p_en import G2p

@st.cache_resource
def setup_nltk(): 
    nltk.download('punkt')
    nltk.download('punkt_tab')

@st.cache_resource
def load_linguistic_model():
    # Cache the neural network so Streamlit doesn't reload it on every button click
    return G2p()

setup_nltk()
with st.spinner("Loading neural network..."):
    g2p_model = load_linguistic_model()

# --- 2. Linguistic Data & Dictionaries ---
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

# Expandable dictionary for context-dependent pronunciations. 
# Matches the NLTK Part of Speech (POS) tag prefix to the correct alignment array.
HETERONYM_RULES = {
    "read": {
        "VBD": [['r', 'R'], ['e', 'EH'], ['a', ''], ['d', 'D']], 
        "VBN": [['r', 'R'], ['e', 'EH'], ['a', ''], ['d', 'D']], 
        "DEFAULT": [['r', 'R'], ['e', 'IY'], ['a', ''], ['d', 'D']]
    },
    "record": {
        "VB": [['r', 'R'], ['e', 'IH'], ['c', 'K'], ['o', 'AO'], ['r', 'R'], ['d', 'D']], 
        "DEFAULT": [['r', 'R'], ['e', 'EH'], ['c', 'K'], ['o', 'ER'], ['r', 'R'], ['d', 'D']]
    },
    "object": {
        "VB": [['o', 'AH'], ['b', 'B'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']], 
        "DEFAULT": [['o', 'AA'], ['b', 'B'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']]
    },
    "tear": {
        "VB": [['t', 'T'], ['e', 'EH'], ['a', ''], ['r', 'R']], 
        "DEFAULT": [['t', 'T'], ['e', 'IY'], ['a', ''], ['r', 'R']]
    },
    "live": {
        "VB": [['l', 'L'], ['i', 'IH'], ['v', 'V'], ['e', '']], 
        "DEFAULT": [['l', 'L'], ['i', 'AY'], ['v', 'V'], ['e', '']]
    },
    "lead": {
        "NN": [['l', 'L'], ['e', 'EH'], ['a', ''], ['d', 'D']], 
        "DEFAULT": [['l', 'L'], ['e', 'IY'], ['a', ''], ['d', 'D']]
    },
    "present": {
        "VB": [['p', 'P'], ['r', 'R'], ['e', 'IY'], ['s', 'Z'], ['e', 'EH'], ['n', 'N'], ['t', 'T']], 
        "DEFAULT": [['p', 'P'], ['r', 'R'], ['e', 'EH'], ['s', 'Z'], ['e', 'AH'], ['n', 'N'], ['t', 'T']]
    },
    "project": {
        "VB": [['p', 'P'], ['r', 'R'], ['o', 'AH'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']], 
        "DEFAULT": [['p', 'P'], ['r', 'R'], ['o', 'AA'], ['j', 'JH'], ['e', 'EH'], ['c', 'K'], ['t', 'T']]
    },
    "wind": {
        "VB": [['w', 'W'], ['i', 'AY'], ['n', 'N'], ['d', 'D']], 
        "DEFAULT": [['w', 'W'], ['i', 'IH'], ['n', 'N'], ['d', 'D']]
    },
    "minute": {
        "JJ": [['m', 'M'], ['i', 'AY'], ['n', 'N'], ['u', 'UW'], ['t', 'T'], ['e', '']], 
        "DEFAULT": [['m', 'M'], ['i', 'IH'], ['n', 'N'], ['u', 'AH'], ['t', 'T'], ['e', '']]
    },
    "use": {
        "VB": [['u', 'Y UW'], ['s', 'Z'], ['e', '']], 
        "DEFAULT": [['u', 'Y UW'], ['s', 'S'], ['e', '']]
    },
    "close": {
        "VB": [['c', 'K'], ['l', 'L'], ['o', 'OW'], ['s', 'Z'], ['e', '']], 
        "DEFAULT": [['c', 'K'], ['l', 'L'], ['o', 'OW'], ['s', 'S'], ['e', '']]
    },
    "resume": {
        "VB": [['r', 'R'], ['e', 'IH'], ['s', 'Z'], ['u', 'UW'], ['m', 'M'], ['e', '']], 
        "DEFAULT": [['r', 'R'], ['e', 'EH'], ['s', 'Z'], ['u', 'UW'], ['m', 'M'], ['e', 'EY']]
    },
    "bow": {
        "VB": [['b', 'B'], ['o', 'AW'], ['w', '']], 
        "DEFAULT": [['b', 'B'], ['o', 'OW'], ['w', '']]
    }
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
    # 1. Ask the neural net to predict phonemes for the entire text at once to retain context
    predicted_output = g2p_model(text_input)
    
    # predicted_output is a flat list (e.g., ['T', 'UH1', 'K', ' ', 'AH0', ' ', 'B', 'AW1'])
    # We parse this into a list of phoneme arrays for each word
    g2p_word_phonemes = []
    current_phonemes = []
    
    for item in predicted_output:
        if item == ' ':
            if current_phonemes:
                g2p_word_phonemes.append(current_phonemes)
                current_phonemes = []
        elif item.isalnum(): 
            # It's a phoneme, so we strip the stress numbers (e.g., AW1 -> AW)
            current_phonemes.append(re.sub(r'\d+', '', item)) 
    if current_phonemes:
        g2p_word_phonemes.append(current_phonemes)
        
    # 2. Tokenize the input to get the original words and sync them with the ML output
    words = nltk.word_tokenize(text_input)
    highlighted_output = []
    word_idx = 0
    
    for word in words:
        # Pass pure punctuation straight to the final output
        if not word.isalnum() and word not in ["n't", "'re", "'ve", "'ll", "'m", "'d", "'s"]:
            highlighted_output.append(word)
            continue
            
        lower_word = word.lower()
        
        # Grab the context-aware phonemes the ML model generated for this specific word
        target_phonemes = g2p_word_phonemes[word_idx] if word_idx < len(g2p_word_phonemes) else []
        word_idx += 1
        
        # 3. Your Dynamic Alignment Engine (repurposed to align ML output on the fly)
        word_alignment = []
        g_idx = 0
        p_idx = 0
        
        while g_idx < len(lower_word) or p_idx < len(target_phonemes):
            g = lower_word[g_idx] if g_idx < len(lower_word) else ""
            p_current = target_phonemes[p_idx] if p_idx < len(target_phonemes) else ""
            p_next = target_phonemes[p_idx+1] if p_idx + 1 < len(target_phonemes) else ""

            if g == 'x' and p_next and p_current in ['K', 'G']:
                word_alignment.append([g, target_phonemes[p_idx] + " " + target_phonemes[p_idx+1]])
                g_idx += 1; p_idx += 2; continue
            if g == 'u' and p_next and p_current == 'Y' and 'UW' in p_next:
                word_alignment.append([g, target_phonemes[p_idx] + " " + target_phonemes[p_idx+1]])
                g_idx += 1; p_idx += 2; continue
            if g == 'o' and p_next and p_current == 'W' and 'AH' in p_next:
                word_alignment.append([g, target_phonemes[p_idx] + " " + target_phonemes[p_idx+1]])
                g_idx += 1; p_idx += 2; continue

            if g_idx < len(lower_word) and p_idx < len(target_phonemes):
                word_alignment.append([g, target_phonemes[p_idx]])
                g_idx += 1; p_idx += 1
            elif g_idx < len(lower_word):
                word_alignment.append([g, ""])
                g_idx += 1
            elif p_idx < len(target_phonemes) and len(word_alignment) > 0:
                word_alignment[-1][1] += " " + target_phonemes[p_idx]
                p_idx += 1
            else:
                break
                
        # 4. Apply Your Multi-Letter Highlight Rules
        highlights = [False] * len(word_alignment)
        
        for i, (g, p) in enumerate(word_alignment):
            if target_phoneme in p.split(): highlights[i] = True
            
        tetraph_rules = {"tion": ["SH","AH","N"], "sion": ["SH","ZH","AH","N"], "eigh": ["EY"], "augh": ["AO","F"], "ough": ["OW","AW","UW","AO","F","AH"]}
        trigraph_rules = {"igh": ["AY"], "tch": ["CH"], "dge": ["JH"], "eau": ["OW","UW"], "ous": ["AH","S"], "que": ["K"]}
        pair_rules = {"sh":["SH"],"ch":["CH","K","SH"],"th":["TH","DH"],"ph":["F"],"wh":["W","HH"],"ng":["NG"],"gh":["F","G"],"ck":["K"],"kn":["N"],"wr":["R"],"mb":["M"],"gn":["N"],"rh":["R"],"ti":["SH"],"ci":["SH"],"si":["SH","ZH"],"ce":["SH"],"tu":["CH"],"su":["SH","ZH"],"ea":["IY","EH","EY"],"ee":["IY"],"oa":["OW"],"oo":["UW","UH"],"ou":["AW","AH","UW","OW"],"ow":["AW","OW"],"ai":["EY","EH"],"ay":["EY"],"ei":["EY","IY"],"ey":["EY","IY"],"au":["AO"],"aw":["AO"],"ew":["UW","Y"],"oe":["OW","UW"],"ie":["IY","AY"],"ui":["UW","IH"],"ue":["UW"]}

        for i in range(len(word_alignment) - 3):
            quad = "".join([a[0] for a in word_alignment[i:i+4]])
            if quad in tetraph_rules and target_phoneme in tetraph_rules[quad]:
                if any(highlights[i:i+4]): highlights[i:i+4] = [True, True, True, True]

        for i in range(len(word_alignment) - 2):
            triple = "".join([a[0] for a in word_alignment[i:i+3]])
            if triple in trigraph_rules and target_phoneme in trigraph_rules[triple]:
                if any(highlights[i:i+3]): highlights[i:i+3] = [True, True, True]

        for i in range(len(word_alignment) - 1):
            pair = "".join([a[0] for a in word_alignment[i:i+2]])
            is_double = (word_alignment[i][0] == word_alignment[i+1][0] and word_alignment[i][0].isalpha())
            if (pair in pair_rules and target_phoneme in pair_rules[pair]) or is_double:
                if any(highlights[i:i+2]): highlights[i:i+2] = [True, True]

        word_html = "".join([f"<span style='background-color: #FFFF00; font-weight: bold; color: black; padding: 0 2px; border-radius: 3px;'>{g}</span>" if highlights[i] else g for i, (g, p) in enumerate(word_alignment)])
        highlighted_output.append(word_html)

    final_html = re.sub(r' ([.,!?\'])', r'\1', " ".join(highlighted_output))
    st.markdown("### Result:")
    st.markdown(f"<div style='font-size: 24px; line-height: 1.5;'>{final_html}</div>", unsafe_allow_html=True)
