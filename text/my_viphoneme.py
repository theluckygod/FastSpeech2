from viphoneme import vi2IPA_split

consonants = ['tʰw', 'ŋ͡m', 'k͡p', 'cw', 'jw', 'bw', 'vw', 'ʈw', 'ʂw', 'fw', 'tʰ', 'tʃ', 'xw', 'ŋw', 'dw', 'ɣw', 'zw', 'mw', 'hw', 'lw', 'kw', 'nw', 't∫', 'ɲw', 'sw', 'tw', 'ʐw', 'dʒ', 'ɲ', 'θ', 'l', 'w', 'd', '∫', 'p', 'ɣ', '!', 'ð', 'ʧ', 'ʒ', 'ʐ', 'z', 'v', 'g', '_', 'ʤ', '.', 'b', 'h', 'n', 'ʂ', 'k', 'm', ' ', 'c', 'j', 'x', 'ʈ', ',', 's', 'ŋ', 'ʃ', '?', 'r', ':', 'η', 'f', ';', 't', "'"]
vowels = ['oʊ', 'ɯəj', 'ɤ̆j', 'ʷiə', 'ɤ̆w', 'ɯəw', 'ʷet', 'iəw', 'uəj', 'ʷen', 'ʷɤ̆', 'ʷiu', 'kwi', 'uə', 'eə', 'oj', 'ʷi', 'ăw', 'aʊ', 'ɛu', 'ɔɪ', 'ʷɤ', 'ɤ̆', 'ʊə', 'zi', 'ʷă', 'eɪ', 'aɪ', 'ew', 'iə', 'ɯj', 'ʷɛ', 'ɯw', 'ɤj', 'ɔ:', 'əʊ', 'ʷa', 'ɑ:', 'ɔj', 'uj', 'ɪə', 'ăj', 'u:', 'aw', 'ɛj', 'iw', 'aj', 'ɜ:', 'eo', 'iɛ', 'ʷe', 'i:', 'ɯə', 'ʌ', 'ɪ', 'ɯ', 'ə', 'u', 'o', 'ă', 'æ', 'ɤ', 'i', 'ɒ', 'ɔ', 'ɛ', 'ʊ', 'a', 'e']
tones = ['1', '3', '6', '2', '5', '4']

def _is_concat_2_vowels(v1, v2):
    if v1 + v2 in vowels:
        return True
    return False

def get_all_syms():
    all_vowels = []
    for v in vowels:
        for t in tones:
            all_vowels.append(v + t)
    return consonants + all_vowels

def get_my_viphoneme_list(ph_list):
    #Returns my viphoneme in list. ex. ['t', 'o', 'j', '5'] -> ['t', 'o5', 'j']

    #Parameters:
    #   ph_list (list): phonemes of words which is parsed by viphoneme

    #Returns:
    #   my_ph (list): my style phonemes

    keep_vowel = False
    should_removed_list = []
    for i in range(len(ph_list)):
        if ph_list[i] in vowels: # find vowels
            if keep_vowel == True:
                if i_keep == i - 1 and _is_concat_2_vowels(ph_list[i_keep], ph_list[i]): # concat 2 vowels
                    ph_list[i_keep] += ph_list[i]
                    should_removed_list.append(i)
                else:
                    ph_list[i_keep] += '1'
                    i_keep = i
            else: # find vowels
                keep_vowel = True
                i_keep = i
        elif ph_list[i] in tones and keep_vowel == True: # concat tone to vowel
            ph_list[i_keep] += ph_list[i]
            should_removed_list.append(i)
            keep_vowel = False
        elif ph_list[i] in ["'", " ", "_"]: # remove special symbol
            should_removed_list.append(i)

    if keep_vowel == True:
        ph_list[i_keep] += '1'

    for i in should_removed_list[::-1]:
        ph_list.pop(i)

    return ph_list

def get_cleaned_viphoneme_list(sentence):
    #Returns cleaned viphoneme in list. ex. "tôi là ai" -> ['t', 'o', 'j', '1', ' ', 'l', 'a', '2', ' ', 'a', 'j', '1']

    #Parameters:
    #   sentence (str): a sentence

    #Returns:
    #   cleaned_ph (list): a list of phonemes which have been cleaned
    cleaned_phs = vi2IPA_split(sentence, '/')[:-9].split('/')
    if len(cleaned_phs) > 0 and cleaned_phs[0] == '':
        cleaned_phs = cleaned_phs[1:]
    return cleaned_phs
