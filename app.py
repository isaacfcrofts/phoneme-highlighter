# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:06:42 2026

@author: Endor
"""
import streamlit as st
import nltk
import re
import urllib.request

# --- 1. Setup & Hybrid Dictionary Builder ---
@st.cache_resource
def setup_nltk_v3(): # Cache busted
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('cmudict') # Added the raw dictionary fallback

@st.cache_data
def build_cloud_dictionary_v7(): # Renamed to v7 to force fresh build
    temp_dict = {}
    
    # PHASE 1: Load the pristine .align file for perfect 1-to-1 matches
    url = "https://raw.githubusercontent.com/kastnerkyle/diphone_synthesizer/master/cmudict.0.7a_SPHINX_40.align"
    try:
        response = urllib.request.urlopen(url)
        lines = response.read().decode('utf-8').splitlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'): continue
            
            tokens = line.split()
            if len(tokens) < 2: continue
                
            raw_word = tokens[0].lower()
            if not raw_word[0].isalpha(): continue
                
            word = raw_word.split('(')[0]
            phonemes = tokens[1:] 

            # The .align file uses underscores to perfectly space out sounds.
            if len(word) == len(phonemes):
                # --- NEW: Edge Case Interceptor ---
                # Skip words where a single letter makes two sounds (o->W+AH, u->Y+UW, x->K+S)
                # so that they safely fall through to your Phase 2 dynamic engine instead.
                if ('o' in word and 'W' in phonemes and 'AH' in phonemes) or \
                   ('u' in word and 'Y' in phonemes and 'UW' in phonemes) or \
                   ('x' in word and (('K' in phonemes and 'S' in phonemes) or ('G' in phonemes and 'Z' in phonemes))):
                    continue
                # ----------------------------------
                
                alignment = []
                for g, p in zip(word, phonemes):
                    p_clean = p if p != '_' else ''
                    alignment.append([g, p_clean])
                if word not in temp_dict:
                    temp_dict[word] = alignment
    except Exception as e:
        st.warning(f"Cloud align file failed: {e}")

    # PHASE 2: Fallback to NLTK raw CMU dict for dropped complex words
    try:
        from nltk.corpus import cmudict
        raw_cmu = cmudict.dict()
        
        for word, pronunciations in raw_cmu.items():
            if word not in temp_dict and word.isalpha():
                phonemes = pronunciations[0] # Take primary pronunciation
                
                # --- DYNAMIC ALIGNMENT ENGINE ---
                word_alignment = []
                g_idx = 0
                p_idx = 0
                
                while g_idx < len(word) or p_idx < len(phonemes):
                    g = word[g_idx] if g_idx < len(word) else ""
                    
                    # Strip numbers from phonemes for accurate logical matching
                    p_current = ''.join([c for c in phonemes[p_idx] if not c.isdigit()]) if p_idx < len(phonemes) else ""
                    p_next = ''.join([c for c in phonemes[p_idx+1] if not c.isdigit()]) if p_idx + 1 < len(phonemes) else ""

                    # 1. Handle 'x' (1 letter -> 2 sounds: K S or G Z)
                    if g == 'x' and p_next and p_current in ['K', 'G']:
                        word_alignment.append([g, phonemes[p_idx] + " " + phonemes[p_idx+1]])
                        g_idx += 1
                        p_idx += 2
                        continue
                        
                    # 2. Handle 'u' making "Y UW" (e.g., music, use)
                    if g == 'u' and p_next and p_current == 'Y' and 'UW' in p_next:
                        word_alignment.append([g, phonemes[p_idx] + " " + phonemes[p_idx+1]])
                        g_idx += 1
                        p_idx += 2
                        continue
                        
                    # 3. Handle 'o' making "W AH" (e.g., once, one)
                    if g == 'o' and p_next and p_current == 'W' and 'AH' in p_next:
                        word_alignment.append([g, phonemes[p_idx] + " " + phonemes[p_idx+1]])
                        g_idx += 1
                        p_idx += 2
                        continue

                    # 4. Standard 1-to-1 match
                    if g_idx < len(word) and p_idx < len(phonemes):
                        word_alignment.append([g, phonemes[p_idx]])
                        g_idx += 1
                        p_idx += 1
                        
                    # 5. Out of sounds, but still have letters (e.g., silent 'e' at the end)
                    elif g_idx < len(word):
                        word_alignment.append([g, ""])
                        g_idx += 1
                        
                    # 6. Out of letters, but still have sounds (Pack extras into the last letter)
                    elif p_idx < len(phonemes) and len(word_alignment) > 0:
                        word_alignment[-1][1] += " " + phonemes[p_idx]
                        p_idx += 1
                    else:
                        break

                temp_dict[word] = word_alignment
                
    except Exception as e:
        st.warning(f"NLTK Fallback failed: {e}")

    return temp_dict

setup_nltk_v3()
with st.spinner("Initializing linguistic engine..."):
    aligned_dict = build_cloud_dictionary_v7()

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
    words = nltk.word_tokenize(text_input)
    tagged_words = nltk.pos_tag(words)
    highlighted_output = []
    
    last_phoneme = None
    
    contraction_rules = {
        "n't": [['n', 'N'], ['\'', ''], ['t', 'T']],
        "'re": [['\'', ''], ['r', 'R'], ['e', '']],
        "'ve": [['\'', ''], ['v', 'V'], ['e', '']],
        "'ll": [['\'', ''], ['l', 'L'], ['l', '']],
        "'m": [['\'', ''], ['m', 'M']],
        "'d": [['\'', ''], ['d', 'D']]
    }

    for word, pos_tag in tagged_words:
        lower_word = word.lower()
        alignment = None
        
        # 1. Intercept punctuation-heavy suffixes before the isalnum() check
        if lower_word == "'s":
            s_sound = 'S' if last_phoneme in ['P', 'T', 'K', 'F', 'TH'] else 'Z'
            alignment = [['\'', ''], ['s', s_sound]]
        elif lower_word in contraction_rules:
            alignment = list(contraction_rules[lower_word])
            
        # 2. Skip other pure punctuation
        if not word.isalnum() and not alignment:
            highlighted_output.append(word)
            continue
            
        # 3. Process dictionary words and intercepted contractions
        if alignment or lower_word in aligned_dict:
            if not alignment:
                # --- NEW: Dictionary-Driven Heteronym Lookup ---
                if lower_word in HETERONYM_RULES:
                    rules = HETERONYM_RULES[lower_word]
                    matched = False
                    # Search for a matching POS prefix (e.g. 'VB' matches 'VBZ', 'VBD', etc.)
                    for tag_key, align_array in rules.items():
                        if tag_key != "DEFAULT" and pos_tag.startswith(tag_key):
                            alignment = list(align_array)
                            matched = True
                            break
                    # Fallback to the default pronunciation if the grammar tag wasn't explicitly caught
                    if not matched and "DEFAULT" in rules:
                        alignment = list(rules["DEFAULT"])
                else:
                    alignment = list(aligned_dict[lower_word])
                # ---------------------------------------------
            
            highlights = [False] * len(alignment)
            
            # 4. Base Matches
            for i, (g, p) in enumerate(alignment):
                if target_phoneme in re.sub(r'\d+', '', p).split():
                    highlights[i] = True
            
            # 5. Multi-Letter Catcher Logic
            tetraph_rules = {"tion": ["SH","AH","N"], "sion": ["SH","ZH","AH","N"], "eigh": ["EY"], "augh": ["AO","F"], "ough": ["OW","AW","UW","AO","F","AH"]}
            trigraph_rules = {"igh": ["AY"], "tch": ["CH"], "dge": ["JH"], "eau": ["OW","UW"], "ous": ["AH","S"], "que": ["K"]}
            pair_rules = {"sh":["SH"],"ch":["CH","K","SH"],"th":["TH","DH"],"ph":["F"],"wh":["W","HH"],"ng":["NG"],"gh":["F","G"],"ck":["K"],"kn":["N"],"wr":["R"],"mb":["M"],"gn":["N"],"rh":["R"],"ti":["SH"],"ci":["SH"],"si":["SH","ZH"],"ce":["SH"],"tu":["CH"],"su":["SH","ZH"],"ea":["IY","EH","EY"],"ee":["IY"],"oa":["OW"],"oo":["UW","UH"],"ou":["AW","AH","UW","OW"],"ow":["AW","OW"],"ai":["EY","EH"],"ay":["EY"],"ei":["EY","IY"],"ey":["EY","IY"],"au":["AO"],"aw":["AO"],"ew":["UW","Y"],"oe":["OW","UW"],"ie":["IY","AY"],"ui":["UW","IH"],"ue":["UW"]}

            for i in range(len(alignment) - 3):
                quad = "".join([a[0] for a in alignment[i:i+4]])
                if quad in tetraph_rules and target_phoneme in tetraph_rules[quad]:
                    if any(highlights[i:i+4]): highlights[i:i+4] = [True, True, True, True]

            for i in range(len(alignment) - 2):
                triple = "".join([a[0] for a in alignment[i:i+3]])
                if triple in trigraph_rules and target_phoneme in trigraph_rules[triple]:
                    if any(highlights[i:i+3]): highlights[i:i+3] = [True, True, True]

            for i in range(len(alignment) - 1):
                pair = "".join([a[0] for a in alignment[i:i+2]])
                is_double = (alignment[i][0] == alignment[i+1][0] and alignment[i][0].isalpha())
                if (pair in pair_rules and target_phoneme in pair_rules[pair]) or is_double:
                    if any(highlights[i:i+2]): highlights[i:i+2] = [True, True]

            # 6. Final Render & State Update
            word_html = "".join([f"<span style='background-color: #FFFF00; font-weight: bold; color: black; padding: 0 2px; border-radius: 3px;'>{g}</span>" if highlights[i] else g for i, (g, p) in enumerate(alignment)])
            highlighted_output.append(word_html)
            
            for g, p in reversed(alignment):
                clean_p = re.sub(r'\d+', '', p).strip()
                if clean_p:
                    last_phoneme = clean_p.split()[-1]
                    break
        else:
            highlighted_output.append(word)

    final_html = re.sub(r' ([.,!?\'])', r'\1', " ".join(highlighted_output))
    st.markdown("### Result:")
    st.markdown(f"<div style='font-size: 24px; line-height: 1.5;'>{final_html}</div>", unsafe_allow_html=True)
