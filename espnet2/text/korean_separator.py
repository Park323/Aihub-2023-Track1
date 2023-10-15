#-*-coding:utf-8-*-

import re

# UNICODE KOR
# Start index : 44032, End index : 55203
BASE_CODE = 44032
JONGSEONG_BASE_CODE = 4520
CHOSEONG_INTV, JUNGSEONG_INTV, JONGSEONG_INTV = 588, 28, 1

# Initial consonants. 19 in total.
CHOSEONG = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# Vowels. 21 in total.
JUNGSEONG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# Final consonants. 1 + 28 in total. (The first token is empty, not using a final consonant)
JONGSEONG = ['\u1160', *[chr(i) for i in range(JONGSEONG_BASE_CODE, JONGSEONG_BASE_CODE + 27)]] # e.g. 'ᆨ', 'ᆩ', ..., 'ᇂ'


def char2grp(sentence):
    splited_chars = list(sentence)

    result = list()
    for char in splited_chars:
        # For Korean character
        if re.match('.*[가-힣]+.*', char) is not None:
            char_code = ord(char) - BASE_CODE
            char1 = int(char_code / CHOSEONG_INTV)
            result.append(CHOSEONG[char1])
            char2 = int((char_code - (CHOSEONG_INTV * char1)) / JUNGSEONG_INTV)
            result.append(JUNGSEONG[char2])
            char3 = int((char_code - (CHOSEONG_INTV * char1) - (JUNGSEONG_INTV * char2)))
            if char3==0:
                pass
            else:
                result.append(JONGSEONG[char3])
        # Keep other characters, numbers, and whatever.
        else:
            result.append(char)
    
    grp_sentence = "".join(result)
    return grp_sentence


def grp2char(sentence):

    grps = list(sentence)
    char_sentence = ""
    
    chr_id = BASE_CODE
    chr_step = 0
    while grps:
        grp = grps.pop(0)
        
        if grp in CHOSEONG:
            if chr_step == 0:
                pass
            elif chr_step == 1: # Ignore previous character (Invalid)
                chr_id = BASE_CODE
                chr_step = 0
            elif chr_step == 2: # End previous character
                char_sentence += chr(chr_id)
                chr_id = BASE_CODE
                chr_step = 0
            else: # Invalid step
                raise Exception
            
            chr_id += CHOSEONG.index(grp) * CHOSEONG_INTV # Initialize new character
            chr_step += 1
        
        elif grp in JUNGSEONG:
            if chr_step == 0: # Ignore this grapheme, keep previous character (Invalid)
                continue
            elif chr_step == 1: # Build character
                chr_id += JUNGSEONG.index(grp) * JUNGSEONG_INTV
                chr_step += 1
            elif chr_step == 2: # Ignore this grapheme, keep previous character (Invalid)
                continue
            else: # Invalid step
                raise Exception
        
        elif grp in JONGSEONG:
            if chr_step == 0: # Ignore this grapheme, keep previous character (Invalid)
                continue
            elif chr_step == 1: # Ignore this grapheme, keep previous character (Invalid)
                continue
            elif chr_step == 2: # Build and end the character
                chr_id += JONGSEONG.index(grp)
                char_sentence += chr(chr_id)
                chr_id = BASE_CODE
                chr_step = 0
            else: # Invalid step
                raise Exception
            
        else:
            if chr_step > 0:
                char_sentence += chr(chr_id)
            char_sentence += grp
            chr_id = BASE_CODE
            chr_step = 0
    
    # Check unfinished char (Choseong + Jungseong)
    if chr_step == 2:
        char_sentence += chr(chr_id)
            
    return char_sentence
