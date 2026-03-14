#!/usr/bin/env python3
# 01_dataset_gen.py
# Run: python 01_dataset_gen.py
# Output: vanimitra_train.jsonl (800+ examples, UTF-8, ensure_ascii=False)

import json
import random

SYSTEM_PROMPT = """You are a voice command parser for a smartphone assistant.
The user speaks in Hindi, Tamil, or English (or mixed/code-switched).
Output ONLY a single valid JSON object with 'intent' and 'params'.
Valid intents: FLASHLIGHT_ON, FLASHLIGHT_OFF, VOLUME_UP, VOLUME_DOWN,
CALL_CONTACT, SET_ALARM, SEND_WHATSAPP, OPEN_APP, NAVIGATE,
TOGGLE_WIFI, TOGGLE_BLUETOOTH, GO_HOME, GO_BACK, LOCK_SCREEN,
TAKE_SCREENSHOT, UNKNOWN"""

def make(user_text, intent, params=None):
    if params is None:
        params = {}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps({"intent": intent, "params": params}, ensure_ascii=False)}
        ]
    }

examples = []

# ─────────────────────────────────────────────
# HINDI — 250 examples
# ─────────────────────────────────────────────

# FLASHLIGHT_ON (15)
examples += [
    make("टॉर्च जलाओ", "FLASHLIGHT_ON"),
    make("टॉर्च चालू करो", "FLASHLIGHT_ON"),
    make("लाइट जलाओ", "FLASHLIGHT_ON"),
    make("टॉर्च ऑन करो", "FLASHLIGHT_ON"),
    make("फ्लैशलाइट जलाओ", "FLASHLIGHT_ON"),
    make("टॉर्च चालू कर दो", "FLASHLIGHT_ON"),
    make("अंधेरा है टॉर्च जलाओ", "FLASHLIGHT_ON"),
    make("जल्दी टॉर्च ऑन करो", "FLASHLIGHT_ON"),
    make("टॉर्च जलाओ यार", "FLASHLIGHT_ON"),
    make("लाइट ऑन करो", "FLASHLIGHT_ON"),
    make("टॉर्च चाहिए", "FLASHLIGHT_ON"),
    make("फ्लैश लाइट चालू करो", "FLASHLIGHT_ON"),
    make("रोशनी चाहिए टॉर्च जलाओ", "FLASHLIGHT_ON"),
    make("torch jalao", "FLASHLIGHT_ON"),
    make("टॉर्च जला दो", "FLASHLIGHT_ON"),
]

# FLASHLIGHT_OFF (12)
examples += [
    make("टॉर्च बंद करो", "FLASHLIGHT_OFF"),
    make("टॉर्च ऑफ करो", "FLASHLIGHT_OFF"),
    make("लाइट बंद करो", "FLASHLIGHT_OFF"),
    make("फ्लैशलाइट बंद करो", "FLASHLIGHT_OFF"),
    make("टॉर्च बुझाओ", "FLASHLIGHT_OFF"),
    make("टॉर्च बंद कर दो", "FLASHLIGHT_OFF"),
    make("लाइट ऑफ करो", "FLASHLIGHT_OFF"),
    make("फ्लैश बंद करो", "FLASHLIGHT_OFF"),
    make("टॉर्च बंद हो जाओ", "FLASHLIGHT_OFF"),
    make("रोशनी बंद करो", "FLASHLIGHT_OFF"),
    make("torch band karo", "FLASHLIGHT_OFF"),
    make("टॉर्च बंद कर", "FLASHLIGHT_OFF"),
]

# VOLUME_UP (12)
examples += [
    make("आवाज़ बढ़ाओ", "VOLUME_UP"),
    make("वॉल्यूम बढ़ाओ", "VOLUME_UP"),
    make("आवाज़ तेज करो", "VOLUME_UP"),
    make("साउंड बढ़ाओ", "VOLUME_UP"),
    make("आवाज़ ऊपर करो", "VOLUME_UP"),
    make("वॉल्यूम ऊपर करो", "VOLUME_UP"),
    make("आवाज़ ज़्यादा करो", "VOLUME_UP"),
    make("तेज़ करो आवाज़", "VOLUME_UP"),
    make("थोड़ा तेज़ करो", "VOLUME_UP"),
    make("volume badao", "VOLUME_UP"),
    make("आवाज बढाओ", "VOLUME_UP"),
    make("ज्यादा आवाज चाहिए", "VOLUME_UP"),
]

# VOLUME_DOWN (12)
examples += [
    make("आवाज़ कम करो", "VOLUME_DOWN"),
    make("वॉल्यूम कम करो", "VOLUME_DOWN"),
    make("आवाज़ धीमी करो", "VOLUME_DOWN"),
    make("साउंड कम करो", "VOLUME_DOWN"),
    make("आवाज़ नीचे करो", "VOLUME_DOWN"),
    make("वॉल्यूम नीचे करो", "VOLUME_DOWN"),
    make("आवाज़ घटाओ", "VOLUME_DOWN"),
    make("थोड़ा धीमा करो", "VOLUME_DOWN"),
    make("आवाज़ बहुत तेज़ है कम करो", "VOLUME_DOWN"),
    make("volume kam karo", "VOLUME_DOWN"),
    make("आवाज कम", "VOLUME_DOWN"),
    make("धीमा करो", "VOLUME_DOWN"),
]

# CALL_CONTACT (20)
examples += [
    make("माँ को कॉल करो", "CALL_CONTACT", {"contact": "माँ"}),
    make("पापा को कॉल करो", "CALL_CONTACT", {"contact": "पापा"}),
    make("राज को कॉल करो", "CALL_CONTACT", {"contact": "राज"}),
    make("अनिता को फोन करो", "CALL_CONTACT", {"contact": "अनिता"}),
    make("भाई को कॉल लगाओ", "CALL_CONTACT", {"contact": "भाई"}),
    make("दीदी को फोन करो", "CALL_CONTACT", {"contact": "दीदी"}),
    make("दोस्त को कॉल करो", "CALL_CONTACT", {"contact": "दोस्त"}),
    make("ऑफिस में कॉल करो", "CALL_CONTACT", {"contact": "ऑफिस"}),
    make("मम्मी को कॉल करो", "CALL_CONTACT", {"contact": "मम्मी"}),
    make("डॉक्टर को फोन करो", "CALL_CONTACT", {"contact": "डॉक्टर"}),
    make("माँ को फोन लगाओ", "CALL_CONTACT", {"contact": "माँ"}),
    make("पापा को फोन करो अभी", "CALL_CONTACT", {"contact": "पापा"}),
    make("राहुल को कॉल करो", "CALL_CONTACT", {"contact": "राहुल"}),
    make("सुनीता को कॉल करो", "CALL_CONTACT", {"contact": "सुनीता"}),
    make("प्रिया को फोन करो", "CALL_CONTACT", {"contact": "प्रिया"}),
    make("नाना को कॉल करो", "CALL_CONTACT", {"contact": "नाना"}),
    make("नानी को फोन करो", "CALL_CONTACT", {"contact": "नानी"}),
    make("बॉस को कॉल करो", "CALL_CONTACT", {"contact": "बॉस"}),
    make("अमित को कॉल लगाओ", "CALL_CONTACT", {"contact": "अमित"}),
    make("पड़ोसी को फोन करो", "CALL_CONTACT", {"contact": "पड़ोसी"}),
]

# SET_ALARM (15)
examples += [
    make("कल सुबह सात बजे अलार्म लगाओ", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("सुबह छह बजे का अलार्म लगाओ", "SET_ALARM", {"time": "06:00"}),
    make("रात आठ बजे अलार्म सेट करो", "SET_ALARM", {"time": "20:00"}),
    make("कल सुबह पाँच बजे उठना है अलार्म लगाओ", "SET_ALARM", {"time": "05:00", "date": "tomorrow"}),
    make("साढ़े सात बजे का अलार्म", "SET_ALARM", {"time": "07:30"}),
    make("अलार्म लगाओ नौ बजे का", "SET_ALARM", {"time": "09:00"}),
    make("सुबह 7 बजे अलार्म", "SET_ALARM", {"time": "07:00"}),
    make("कल के लिए 6 बजे का अलार्म", "SET_ALARM", {"time": "06:00", "date": "tomorrow"}),
    make("10 बजे अलार्म लगाओ", "SET_ALARM", {"time": "10:00"}),
    make("alarm lagao kal subah 7 baje", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("सुबह साढ़े छह बजे अलार्म", "SET_ALARM", {"time": "06:30"}),
    make("रात 11 बजे का अलार्म", "SET_ALARM", {"time": "23:00"}),
    make("कल 8 बजे उठना है", "SET_ALARM", {"time": "08:00", "date": "tomorrow"}),
    make("अलार्म सेट कर सुबह 5:30 बजे", "SET_ALARM", {"time": "05:30"}),
    make("दोपहर 2 बजे अलार्म लगाओ", "SET_ALARM", {"time": "14:00"}),
]

# SEND_WHATSAPP (12)
examples += [
    make("राज को व्हाट्सएप करो मैं देर से आऊंगा", "SEND_WHATSAPP", {"contact": "राज", "message": "मैं देर से आऊंगा"}),
    make("माँ को मैसेज करो आ रहा हूँ", "SEND_WHATSAPP", {"contact": "माँ", "message": "आ रहा हूँ"}),
    make("पापा को व्हाट्सएप भेजो ठीक हूँ", "SEND_WHATSAPP", {"contact": "पापा", "message": "ठीक हूँ"}),
    make("अनिता को लिखो पार्टी में आओ", "SEND_WHATSAPP", {"contact": "अनिता", "message": "पार्टी में आओ"}),
    make("भाई को मैसेज करो कहाँ हो", "SEND_WHATSAPP", {"contact": "भाई", "message": "कहाँ हो"}),
    make("राहुल को व्हाट्सएप करो मीटिंग कैंसिल है", "SEND_WHATSAPP", {"contact": "राहुल", "message": "मीटिंग कैंसिल है"}),
    make("प्रिया को मैसेज भेजो कल मिलते हैं", "SEND_WHATSAPP", {"contact": "प्रिया", "message": "कल मिलते हैं"}),
    make("दीदी को व्हाट्सएप करो घर पर हूँ", "SEND_WHATSAPP", {"contact": "दीदी", "message": "घर पर हूँ"}),
    make("ऑफिस में मैसेज करो लेट हो रहा हूँ", "SEND_WHATSAPP", {"contact": "ऑफिस", "message": "लेट हो रहा हूँ"}),
    make("अमित को लिखो ट्रेन मिस हो गई", "SEND_WHATSAPP", {"contact": "अमित", "message": "ट्रेन मिस हो गई"}),
    make("मम्मी को व्हाट्सएप करो खाना खा लिया", "SEND_WHATSAPP", {"contact": "मम्मी", "message": "खाना खा लिया"}),
    make("बॉस को मैसेज करो आज नहीं आ सकता", "SEND_WHATSAPP", {"contact": "बॉस", "message": "आज नहीं आ सकता"}),
]

# OPEN_APP (14)
examples += [
    make("सेटिंग्स खोलो", "OPEN_APP", {"app": "settings"}),
    make("क्रोम खोलो", "OPEN_APP", {"app": "chrome"}),
    make("यूट्यूब खोलो", "OPEN_APP", {"app": "youtube"}),
    make("कैमरा खोलो", "OPEN_APP", {"app": "camera"}),
    make("मैप्स खोलो", "OPEN_APP", {"app": "maps"}),
    make("व्हाट्सएप खोलो", "OPEN_APP", {"app": "whatsapp"}),
    make("स्पॉटिफाई खोलो", "OPEN_APP", {"app": "spotify"}),
    make("सेटिंग्स ओपन करो", "OPEN_APP", {"app": "settings"}),
    make("यूट्यूब ओपन करो", "OPEN_APP", {"app": "youtube"}),
    make("क्रोम ब्राउज़र खोलो", "OPEN_APP", {"app": "chrome"}),
    make("गूगल मैप्स खोलो", "OPEN_APP", {"app": "maps"}),
    make("कैमरा ऐप खोलो", "OPEN_APP", {"app": "camera"}),
    make("सेटिंग में जाओ", "OPEN_APP", {"app": "settings"}),
    make("spotify kholo", "OPEN_APP", {"app": "spotify"}),
]

# NAVIGATE (10)
examples += [
    make("हवाई अड्डे नेविगेट करो", "NAVIGATE", {"destination": "airport"}),
    make("घर का रास्ता दिखाओ", "NAVIGATE", {"destination": "घर"}),
    make("अस्पताल ले चलो", "NAVIGATE", {"destination": "hospital"}),
    make("रेलवे स्टेशन जाना है", "NAVIGATE", {"destination": "railway station"}),
    make("नजदीकी पेट्रोल पंप तक नेविगेट करो", "NAVIGATE", {"destination": "petrol pump"}),
    make("मॉल तक का रास्ता बताओ", "NAVIGATE", {"destination": "mall"}),
    make("ऑफिस जाना है", "NAVIGATE", {"destination": "office"}),
    make("बस स्टैंड का रास्ता दिखाओ", "NAVIGATE", {"destination": "bus stand"}),
    make("एयरपोर्ट कैसे जाएं", "NAVIGATE", {"destination": "airport"}),
    make("नजदीकी एटीएम बताओ", "NAVIGATE", {"destination": "ATM"}),
]

# TOGGLE_WIFI (8)
examples += [
    make("वाईफाई चालू करो", "TOGGLE_WIFI", {"state": "on"}),
    make("वाईफाई बंद करो", "TOGGLE_WIFI", {"state": "off"}),
    make("वाईफाई ऑन करो", "TOGGLE_WIFI", {"state": "on"}),
    make("वाईफाई ऑफ करो", "TOGGLE_WIFI", {"state": "off"}),
    make("wifi on kar do", "TOGGLE_WIFI", {"state": "on"}),
    make("wifi band karo", "TOGGLE_WIFI", {"state": "off"}),
    make("नेट चालू करो", "TOGGLE_WIFI", {"state": "on"}),
    make("इंटरनेट बंद करो", "TOGGLE_WIFI", {"state": "off"}),
]

# TOGGLE_BLUETOOTH (8)
examples += [
    make("ब्लूटूथ चालू करो", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("ब्लूटूथ बंद करो", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("ब्लूटूथ ऑन करो", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("ब्लूटूथ ऑफ करो", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("bluetooth on karo", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("bluetooth off karo", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("ब्लूटूथ चालू कर दो", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("ब्लूटूथ बंद कर दो", "TOGGLE_BLUETOOTH", {"state": "off"}),
]

# GO_HOME (8)
examples += [
    make("होम जाओ", "GO_HOME"),
    make("होम स्क्रीन पर जाओ", "GO_HOME"),
    make("मुख्य स्क्रीन पर जाओ", "GO_HOME"),
    make("होम पेज पर जाओ", "GO_HOME"),
    make("home screen pe jao", "GO_HOME"),
    make("होम बटन दबाओ", "GO_HOME"),
    make("डेस्कटॉप पर जाओ", "GO_HOME"),
    make("मेन स्क्रीन", "GO_HOME"),
]

# GO_BACK (8)
examples += [
    make("वापस जाओ", "GO_BACK"),
    make("पीछे जाओ", "GO_BACK"),
    make("बैक करो", "GO_BACK"),
    make("पिछले पेज पर जाओ", "GO_BACK"),
    make("वापस", "GO_BACK"),
    make("पीछे", "GO_BACK"),
    make("back jao", "GO_BACK"),
    make("एक कदम पीछे जाओ", "GO_BACK"),
]

# LOCK_SCREEN (7)
examples += [
    make("फोन लॉक करो", "LOCK_SCREEN"),
    make("स्क्रीन लॉक करो", "LOCK_SCREEN"),
    make("फोन बंद करो", "LOCK_SCREEN"),
    make("लॉक कर दो", "LOCK_SCREEN"),
    make("phone lock karo", "LOCK_SCREEN"),
    make("स्क्रीन बंद करो", "LOCK_SCREEN"),
    make("फोन स्लीप मोड में डालो", "LOCK_SCREEN"),
]

# TAKE_SCREENSHOT (7)
examples += [
    make("स्क्रीनशॉट लो", "TAKE_SCREENSHOT"),
    make("स्क्रीनशॉट खींचो", "TAKE_SCREENSHOT"),
    make("स्क्रीन कैप्चर करो", "TAKE_SCREENSHOT"),
    make("screenshot lo", "TAKE_SCREENSHOT"),
    make("स्क्रीन का फोटो लो", "TAKE_SCREENSHOT"),
    make("इस स्क्रीन का स्क्रीनशॉट", "TAKE_SCREENSHOT"),
    make("कैप्चर करो", "TAKE_SCREENSHOT"),
]

# Whisper mishearing variants - Hindi (10)
examples += [
    make("torch jala oh", "FLASHLIGHT_ON"),          # mishear
    make("alarm way early 7 money", "SET_ALARM", {"time": "07:00"}),  # Whisper classic
    make("volume bad how", "VOLUME_UP"),
    make("wahts app karo raj party hay aaj", "SEND_WHATSAPP", {"contact": "raj", "message": "party hay aaj"}),
    make("call karo maa ko", "CALL_CONTACT", {"contact": "maa"}),
    make("you tube kho lo", "OPEN_APP", {"app": "youtube"}),
    make("screen shot lo", "TAKE_SCREENSHOT"),
    make("blue tooth on karo", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("wi fi band karo", "TOGGLE_WIFI", {"state": "off"}),
    make("wapas ja oh", "GO_BACK"),
]

print(f"Hindi examples so far: {len(examples)}")

# ─────────────────────────────────────────────
# TAMIL — 250 examples
# ─────────────────────────────────────────────

tamil_examples = []

# FLASHLIGHT_ON (12)
tamil_examples += [
    make("விளக்கை ஆன் பண்ணு", "FLASHLIGHT_ON"),
    make("டார்ச் ஆன் பண்ணு", "FLASHLIGHT_ON"),
    make("லைட் போடு", "FLASHLIGHT_ON"),
    make("விளக்கு வை", "FLASHLIGHT_ON"),
    make("டார்ச் லைட் ஆன் பண்ணு", "FLASHLIGHT_ON"),
    make("ஃப்ளாஷ் லைட் ஆன் பண்ணு", "FLASHLIGHT_ON"),
    make("விளக்கு ஆன் செய்", "FLASHLIGHT_ON"),
    make("வெளிச்சம் போடு", "FLASHLIGHT_ON"),
    make("லைட் ஆன் பண்ணு", "FLASHLIGHT_ON"),
    make("டார்ச் வை", "FLASHLIGHT_ON"),
    make("விளக்கை எரி", "FLASHLIGHT_ON"),
    make("ஃப்ளாஷ் போடு", "FLASHLIGHT_ON"),
]

# FLASHLIGHT_OFF (10)
tamil_examples += [
    make("விளக்கை ஆஃப் பண்ணு", "FLASHLIGHT_OFF"),
    make("டார்ச் ஆஃப் பண்ணு", "FLASHLIGHT_OFF"),
    make("லைட் அணை", "FLASHLIGHT_OFF"),
    make("விளக்கு அணை", "FLASHLIGHT_OFF"),
    make("டார்ச் லைட் ஆஃப் பண்ணு", "FLASHLIGHT_OFF"),
    make("ஃப்ளாஷ் ஆஃப் பண்ணு", "FLASHLIGHT_OFF"),
    make("விளக்கை நிறுத்து", "FLASHLIGHT_OFF"),
    make("லைட் ஆஃப் செய்", "FLASHLIGHT_OFF"),
    make("டார்ச் போட்டை அணை", "FLASHLIGHT_OFF"),
    make("வெளிச்சம் அணை", "FLASHLIGHT_OFF"),
]

# VOLUME_UP (10)
tamil_examples += [
    make("வால்யூம் கூட்டு", "VOLUME_UP"),
    make("சத்தம் கூட்டு", "VOLUME_UP"),
    make("ஒலி கூட்டு", "VOLUME_UP"),
    make("வால்யூம் அதிகம் பண்ணு", "VOLUME_UP"),
    make("சத்தம் அதிகமாக வை", "VOLUME_UP"),
    make("லவுட் பண்ணு", "VOLUME_UP"),
    make("சத்தம் ஏத்து", "VOLUME_UP"),
    make("கொஞ்சம் சத்தமா வை", "VOLUME_UP"),
    make("வால்யூம் ஏத்து", "VOLUME_UP"),
    make("ஒலியை கூட்டு", "VOLUME_UP"),
]

# VOLUME_DOWN (10)
tamil_examples += [
    make("வால்யூம் குறை", "VOLUME_DOWN"),
    make("சத்தம் குறை", "VOLUME_DOWN"),
    make("ஒலி குறை", "VOLUME_DOWN"),
    make("வால்யூம் கம் பண்ணு", "VOLUME_DOWN"),
    make("சத்தம் குறைவா வை", "VOLUME_DOWN"),
    make("லோ பண்ணு", "VOLUME_DOWN"),
    make("சத்தம் தாழ்த்து", "VOLUME_DOWN"),
    make("கொஞ்சம் குறைவா வை", "VOLUME_DOWN"),
    make("வால்யூம் தாழ்த்து", "VOLUME_DOWN"),
    make("ஒலியை குறை", "VOLUME_DOWN"),
]

# CALL_CONTACT (20)
tamil_examples += [
    make("அம்மாவை கால் பண்ணு", "CALL_CONTACT", {"contact": "அம்மா"}),
    make("அப்பாவை கால் பண்ணு", "CALL_CONTACT", {"contact": "அப்பா"}),
    make("ராஜுவை கால் பண்ணு", "CALL_CONTACT", {"contact": "ராஜு"}),
    make("அக்காவை போன் பண்ணு", "CALL_CONTACT", {"contact": "அக்கா"}),
    make("அண்ணனை கால் பண்ணு", "CALL_CONTACT", {"contact": "அண்ணன்"}),
    make("தம்பியை போன் பண்ணு", "CALL_CONTACT", {"contact": "தம்பி"}),
    make("தங்கையை கால் பண்ணு", "CALL_CONTACT", {"contact": "தங்கை"}),
    make("நண்பனை கால் பண்ணு", "CALL_CONTACT", {"contact": "நண்பன்"}),
    make("டாக்டரை போன் பண்ணு", "CALL_CONTACT", {"contact": "டாக்டர்"}),
    make("ஆஃபீஸுக்கு கால் பண்ணு", "CALL_CONTACT", {"contact": "ஆஃபீஸ்"}),
    make("அம்மாவை போன் பண்ணு", "CALL_CONTACT", {"contact": "அம்மா"}),
    make("அப்பாவை இப்போவே கால் பண்ணு", "CALL_CONTACT", {"contact": "அப்பா"}),
    make("சுரேஷை கால் பண்ணு", "CALL_CONTACT", {"contact": "சுரேஷ்"}),
    make("பிரியாவை போன் பண்ணு", "CALL_CONTACT", {"contact": "பிரியா"}),
    make("மாமாவை கால் பண்ணு", "CALL_CONTACT", {"contact": "மாமா"}),
    make("அத்தையை போன் பண்ணு", "CALL_CONTACT", {"contact": "அத்தை"}),
    make("தாத்தாவை கால் பண்ணு", "CALL_CONTACT", {"contact": "தாத்தா"}),
    make("பாட்டியை போன் பண்ணு", "CALL_CONTACT", {"contact": "பாட்டி"}),
    make("முருகனை கால் பண்ணு", "CALL_CONTACT", {"contact": "முருகன்"}),
    make("லட்சுமியை போன் பண்ணு", "CALL_CONTACT", {"contact": "லட்சுமி"}),
]

# SET_ALARM (15)
tamil_examples += [
    make("நாளை காலை ஏழு மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("காலை ஆறு மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "06:00"}),
    make("இரவு எட்டு மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "20:00"}),
    make("நாளை காலை ஐந்து மணிக்கு எழுந்திருக்கணும் அலாரம் வை", "SET_ALARM", {"time": "05:00", "date": "tomorrow"}),
    make("ஏழரை மணிக்கு அலாரம்", "SET_ALARM", {"time": "07:30"}),
    make("ஒன்பது மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "09:00"}),
    make("காலை 7 மணிக்கு அலாரம்", "SET_ALARM", {"time": "07:00"}),
    make("நாளை 6 மணிக்கு அலாரம்", "SET_ALARM", {"time": "06:00", "date": "tomorrow"}),
    make("10 மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "10:00"}),
    make("alarm vei naalai kaalai 7 mani", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("காலை ஆறரை மணிக்கு அலாரம்", "SET_ALARM", {"time": "06:30"}),
    make("இரவு 11 மணிக்கு அலாரம்", "SET_ALARM", {"time": "23:00"}),
    make("நாளை 8 மணிக்கு எழுந்திருக்கணும்", "SET_ALARM", {"time": "08:00", "date": "tomorrow"}),
    make("காலை 5:30 மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "05:30"}),
    make("மதியம் 2 மணிக்கு அலாரம் வை", "SET_ALARM", {"time": "14:00"}),
]

# SEND_WHATSAPP (12)
tamil_examples += [
    make("ராஜுக்கு வாட்ஸ்அப் பண்ணு நான் லேட்டாக வருவேன்", "SEND_WHATSAPP", {"contact": "ராஜு", "message": "நான் லேட்டாக வருவேன்"}),
    make("அம்மாவுக்கு மெசேஜ் பண்ணு வந்துட்டிருக்கேன்", "SEND_WHATSAPP", {"contact": "அம்மா", "message": "வந்துட்டிருக்கேன்"}),
    make("அப்பாவுக்கு வாட்ஸ்அப் அனுப்பு நலமா இருக்கேன்", "SEND_WHATSAPP", {"contact": "அப்பா", "message": "நலமா இருக்கேன்"}),
    make("பிரியாவுக்கு எழுது பார்ட்டிக்கு வா", "SEND_WHATSAPP", {"contact": "பிரியா", "message": "பார்ட்டிக்கு வா"}),
    make("அண்ணனுக்கு மெசேஜ் பண்ணு எங்க இருக்க", "SEND_WHATSAPP", {"contact": "அண்ணன்", "message": "எங்க இருக்க"}),
    make("சுரேஷுக்கு வாட்ஸ்அப் பண்ணு மீட்டிங் கேன்சல்", "SEND_WHATSAPP", {"contact": "சுரேஷ்", "message": "மீட்டிங் கேன்சல்"}),
    make("அக்காவுக்கு மெசேஜ் அனுப்பு நாளை சந்திப்போம்", "SEND_WHATSAPP", {"contact": "அக்கா", "message": "நாளை சந்திப்போம்"}),
    make("நண்பனுக்கு வாட்ஸ்அப் பண்ணு ரெடியா", "SEND_WHATSAPP", {"contact": "நண்பன்", "message": "ரெடியா"}),
    make("ஆஃபீஸுக்கு மெசேஜ் பண்ணு லேட்டாக வருவேன்", "SEND_WHATSAPP", {"contact": "ஆஃபீஸ்", "message": "லேட்டாக வருவேன்"}),
    make("முருகனுக்கு எழுது ரயில் மிஸ் ஆச்சு", "SEND_WHATSAPP", {"contact": "முருகன்", "message": "ரயில் மிஸ் ஆச்சு"}),
    make("அம்மாவுக்கு வாட்ஸ்அப் சாப்பிட்டேன்", "SEND_WHATSAPP", {"contact": "அம்மா", "message": "சாப்பிட்டேன்"}),
    make("மேனேஜருக்கு மெசேஜ் பண்ணு இன்னைக்கு வர முடியாது", "SEND_WHATSAPP", {"contact": "மேனேஜர்", "message": "இன்னைக்கு வர முடியாது"}),
]

# OPEN_APP (14)
tamil_examples += [
    make("செட்டிங்ஸ் திற", "OPEN_APP", {"app": "settings"}),
    make("யூடியூப் திற", "OPEN_APP", {"app": "youtube"}),
    make("கேமரா திற", "OPEN_APP", {"app": "camera"}),
    make("மேப்ஸ் திற", "OPEN_APP", {"app": "maps"}),
    make("வாட்ஸ்அப் திற", "OPEN_APP", {"app": "whatsapp"}),
    make("ஸ்போட்டிஃபை திற", "OPEN_APP", {"app": "spotify"}),
    make("க்ரோம் திற", "OPEN_APP", {"app": "chrome"}),
    make("செட்டிங்ஸ் ஓபன் பண்ணு", "OPEN_APP", {"app": "settings"}),
    make("யூடியூப் ஓபன் பண்ணு", "OPEN_APP", {"app": "youtube"}),
    make("க்ரோம் திறந்துவிடு", "OPEN_APP", {"app": "chrome"}),
    make("கூகுள் மேப்ஸ் திற", "OPEN_APP", {"app": "maps"}),
    make("கேமரா ஆப் திற", "OPEN_APP", {"app": "camera"}),
    make("செட்டிங்ஸ்ல போ", "OPEN_APP", {"app": "settings"}),
    make("ஸ்போட்டிஃபை ஓபன் பண்ணு", "OPEN_APP", {"app": "spotify"}),
]

# NAVIGATE (10)
tamil_examples += [
    make("விமான நிலையம் போக வழி காட்டு", "NAVIGATE", {"destination": "விமான நிலையம்"}),
    make("வீட்டுக்கு வழி காட்டு", "NAVIGATE", {"destination": "வீடு"}),
    make("மருத்துவமனைக்கு வழி காட்டு", "NAVIGATE", {"destination": "மருத்துவமனை"}),
    make("ரயில் நிலையம் போகணும்", "NAVIGATE", {"destination": "ரயில் நிலையம்"}),
    make("கிட்ட பெட்ரோல் பங்க் வழி காட்டு", "NAVIGATE", {"destination": "petrol bunk"}),
    make("மால் போக வழி காட்டு", "NAVIGATE", {"destination": "mall"}),
    make("ஆஃபீஸ் போகணும்", "NAVIGATE", {"destination": "office"}),
    make("பஸ் ஸ்டாண்ட் வழி காட்டு", "NAVIGATE", {"destination": "bus stand"}),
    make("ஏர்போர்ட் எப்படி போவது", "NAVIGATE", {"destination": "airport"}),
    make("கிட்ட ATM வழி காட்டு", "NAVIGATE", {"destination": "ATM"}),
]

# TOGGLE_WIFI (8)
tamil_examples += [
    make("வைஃபை ஆன் பண்ணு", "TOGGLE_WIFI", {"state": "on"}),
    make("வைஃபை ஆஃப் பண்ணு", "TOGGLE_WIFI", {"state": "off"}),
    make("வைஃபை போடு", "TOGGLE_WIFI", {"state": "on"}),
    make("வைஃபை நிறுத்து", "TOGGLE_WIFI", {"state": "off"}),
    make("இன்டர்நெட் போடு", "TOGGLE_WIFI", {"state": "on"}),
    make("இன்டர்நெட் நிறுத்து", "TOGGLE_WIFI", {"state": "off"}),
    make("நெட் ஆன் பண்ணு", "TOGGLE_WIFI", {"state": "on"}),
    make("நெட் ஆஃப் பண்ணு", "TOGGLE_WIFI", {"state": "off"}),
]

# TOGGLE_BLUETOOTH (8)
tamil_examples += [
    make("ப்ளூடூத் ஆன் பண்ணு", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("ப்ளூடூத் ஆஃப் பண்ணு", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("ப்ளூடூத் போடு", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("ப்ளூடூத் நிறுத்து", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("bluetooth on pannu", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("bluetooth off pannu", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("ப்ளூடூத் ஆன் செய்", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("ப்ளூடூத் ஆஃப் செய்", "TOGGLE_BLUETOOTH", {"state": "off"}),
]

# GO_HOME (8)
tamil_examples += [
    make("ஹோம் போ", "GO_HOME"),
    make("ஹோம் ஸ்க்ரீனுக்கு போ", "GO_HOME"),
    make("முகப்புத் திரைக்கு போ", "GO_HOME"),
    make("ஹோம் பேஜுக்கு போ", "GO_HOME"),
    make("home screen poo", "GO_HOME"),
    make("ஹோம் பட்டன் அழுத்து", "GO_HOME"),
    make("டெஸ்க்டாப்புக்கு போ", "GO_HOME"),
    make("மெயின் ஸ்க்ரீன்", "GO_HOME"),
]

# GO_BACK (8)
tamil_examples += [
    make("திரும்பு போ", "GO_BACK"),
    make("பின்னாடி போ", "GO_BACK"),
    make("பேக் பண்ணு", "GO_BACK"),
    make("முந்தைய பக்கத்துக்கு போ", "GO_BACK"),
    make("திரும்பு", "GO_BACK"),
    make("பின்னாடி", "GO_BACK"),
    make("back pannu", "GO_BACK"),
    make("ஒரு படி பின்னாடி போ", "GO_BACK"),
]

# LOCK_SCREEN (7)
tamil_examples += [
    make("போனை லாக் பண்ணு", "LOCK_SCREEN"),
    make("ஸ்க்ரீனை லாக் பண்ணு", "LOCK_SCREEN"),
    make("போனை மூடு", "LOCK_SCREEN"),
    make("லாக் பண்ணுடா", "LOCK_SCREEN"),
    make("phone lock pannu", "LOCK_SCREEN"),
    make("ஸ்க்ரீன் ஆஃப் பண்ணு", "LOCK_SCREEN"),
    make("ஸ்லீப் மோட்ல போடு", "LOCK_SCREEN"),
]

# TAKE_SCREENSHOT (8)
tamil_examples += [
    make("ஸ்கிரீன்ஷாட் எடு", "TAKE_SCREENSHOT"),
    make("ஸ்க்ரீன்ஷாட் எடு", "TAKE_SCREENSHOT"),
    make("ஸ்க்ரீன் கேப்சர் பண்ணு", "TAKE_SCREENSHOT"),
    make("screenshot edu", "TAKE_SCREENSHOT"),
    make("ஸ்க்ரீன் போட்டோ எடு", "TAKE_SCREENSHOT"),
    make("இந்த ஸ்க்ரீனை ஷாட் எடு", "TAKE_SCREENSHOT"),
    make("கேப்சர் பண்ணு", "TAKE_SCREENSHOT"),
    make("ஸ்க்ரீன்ஷாட் போடு", "TAKE_SCREENSHOT"),
]

# Whisper mishearing variants - Tamil (8)
tamil_examples += [
    make("alaaram vei kaalai 7 mani", "SET_ALARM", {"time": "07:00"}),  # classic Whisper mishear
    make("alarm way early 7 money", "SET_ALARM", {"time": "07:00"}),
    make("volume cootu", "VOLUME_UP"),
    make("amma kaal pannu", "CALL_CONTACT", {"contact": "amma"}),
    make("you tube thira", "OPEN_APP", {"app": "youtube"}),
    make("screen shot edu", "TAKE_SCREENSHOT"),
    make("blue tooth on pannu", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("wi fi off pannu", "TOGGLE_WIFI", {"state": "off"}),
]

examples += tamil_examples
print(f"Tamil examples added. Total so far: {len(examples)}")

# ─────────────────────────────────────────────
# CODE-SWITCHED — 200 examples
# ─────────────────────────────────────────────

cs_examples = []

# FLASHLIGHT (15)
cs_examples += [
    make("torch on pannu", "FLASHLIGHT_ON"),
    make("torch off pannu", "FLASHLIGHT_OFF"),
    make("torch jalao yaar", "FLASHLIGHT_ON"),
    make("torch band karo", "FLASHLIGHT_OFF"),
    make("light on pannu", "FLASHLIGHT_ON"),
    make("light off pannu", "FLASHLIGHT_OFF"),
    make("flashlight on kar do", "FLASHLIGHT_ON"),
    make("flashlight off kar do", "FLASHLIGHT_OFF"),
    make("torch on karo bhai", "FLASHLIGHT_ON"),
    make("andhere mein torch on pannu", "FLASHLIGHT_ON"),
    make("torch aachu off pannu", "FLASHLIGHT_OFF"),
    make("torch venum on pannu", "FLASHLIGHT_ON"),
    make("light venam off kar", "FLASHLIGHT_OFF"),
    make("torch on maadi", "FLASHLIGHT_ON"),
    make("torch off maadi", "FLASHLIGHT_OFF"),
]

# VOLUME (12)
cs_examples += [
    make("volume badhao", "VOLUME_UP"),
    make("volume kam karo", "VOLUME_DOWN"),
    make("volume koodunga", "VOLUME_UP"),
    make("volume kuraikka", "VOLUME_DOWN"),
    make("sound badhao yaar", "VOLUME_UP"),
    make("sound kam kar do", "VOLUME_DOWN"),
    make("aavaaz zyada pannu", "VOLUME_UP"),
    make("aavaaz kam pannu", "VOLUME_DOWN"),
    make("volume up pannu", "VOLUME_UP"),
    make("volume down pannu", "VOLUME_DOWN"),
    make("thoda loud kar do", "VOLUME_UP"),
    make("thoda kam kar do", "VOLUME_DOWN"),
]

# CALL_CONTACT (20)
cs_examples += [
    make("amma ku call pannu", "CALL_CONTACT", {"contact": "amma"}),
    make("appa ko call karo", "CALL_CONTACT", {"contact": "appa"}),
    make("maa ko call pannu", "CALL_CONTACT", {"contact": "maa"}),
    make("papa ko call kar do", "CALL_CONTACT", {"contact": "papa"}),
    make("bhai ko call pannu", "CALL_CONTACT", {"contact": "bhai"}),
    make("akka ku call pannu", "CALL_CONTACT", {"contact": "akka"}),
    make("raj ko call karo", "CALL_CONTACT", {"contact": "raj"}),
    make("raju ku call pannu", "CALL_CONTACT", {"contact": "raju"}),
    make("friend ko call kar do", "CALL_CONTACT", {"contact": "friend"}),
    make("nanban ku call pannu", "CALL_CONTACT", {"contact": "nanban"}),
    make("doctor ko call pannu", "CALL_CONTACT", {"contact": "doctor"}),
    make("office ku call pannu", "CALL_CONTACT", {"contact": "office"}),
    make("didi ko call karo", "CALL_CONTACT", {"contact": "didi"}),
    make("thatha ku call pannu", "CALL_CONTACT", {"contact": "thatha"}),
    make("paatti ku call pannu", "CALL_CONTACT", {"contact": "paatti"}),
    make("mama ku call pannu", "CALL_CONTACT", {"contact": "mama"}),
    make("boss ko call kar", "CALL_CONTACT", {"contact": "boss"}),
    make("amit ko call karo bhai", "CALL_CONTACT", {"contact": "amit"}),
    make("priya ku call pannu", "CALL_CONTACT", {"contact": "priya"}),
    make("suresh ko call karo", "CALL_CONTACT", {"contact": "suresh"}),
]

# SET_ALARM (15)
cs_examples += [
    make("kal subah 7 baje alarm", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("7 mani ku alarm vei", "SET_ALARM", {"time": "07:00"}),
    make("naalai kaalai 6 mani alarm", "SET_ALARM", {"time": "06:00", "date": "tomorrow"}),
    make("subah 5 baje alarm lagao", "SET_ALARM", {"time": "05:00"}),
    make("kal 8 baje uthna hai alarm lagao", "SET_ALARM", {"time": "08:00", "date": "tomorrow"}),
    make("alarm set karo 7:30", "SET_ALARM", {"time": "07:30"}),
    make("6 mani alarm vei naalai", "SET_ALARM", {"time": "06:00", "date": "tomorrow"}),
    make("9 baje ka alarm", "SET_ALARM", {"time": "09:00"}),
    make("10 mani ku alarm", "SET_ALARM", {"time": "10:00"}),
    make("alarm lagao subah 5:30 baje", "SET_ALARM", {"time": "05:30"}),
    make("naalai 7 mani ku ezhunga alarm", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("kal ke liye alarm 6 baje", "SET_ALARM", {"time": "06:00", "date": "tomorrow"}),
    make("alarm vei kal morning 8", "SET_ALARM", {"time": "08:00", "date": "tomorrow"}),
    make("subah 7 baje alarm vei", "SET_ALARM", {"time": "07:00"}),
    make("evening 6 mani alarm", "SET_ALARM", {"time": "18:00"}),
]

# SEND_WHATSAPP (15)
cs_examples += [
    make("WhatsApp karo Raj ko party hai aaj", "SEND_WHATSAPP", {"contact": "Raj", "message": "party hai aaj"}),
    make("amma ku whatsapp pannu vandhutiren", "SEND_WHATSAPP", {"contact": "amma", "message": "vandhutiren"}),
    make("bhai ko message karo late ho raha hun", "SEND_WHATSAPP", {"contact": "bhai", "message": "late ho raha hun"}),
    make("raju ku message pannu ready ah", "SEND_WHATSAPP", {"contact": "raju", "message": "ready ah"}),
    make("priya ko WhatsApp karo kal milenge", "SEND_WHATSAPP", {"contact": "priya", "message": "kal milenge"}),
    make("office ku message pannu late aaven", "SEND_WHATSAPP", {"contact": "office", "message": "late aaven"}),
    make("papa ko message kar do theek hun", "SEND_WHATSAPP", {"contact": "papa", "message": "theek hun"}),
    make("maa ko WhatsApp pannu aa raha hun", "SEND_WHATSAPP", {"contact": "maa", "message": "aa raha hun"}),
    make("suresh ku whatsapp pannu meeting cancel", "SEND_WHATSAPP", {"contact": "suresh", "message": "meeting cancel"}),
    make("raj ko likh do train miss ho gayi", "SEND_WHATSAPP", {"contact": "raj", "message": "train miss ho gayi"}),
    make("akka ku message pannu naalai santhippom", "SEND_WHATSAPP", {"contact": "akka", "message": "naalai santhippom"}),
    make("boss ku whatsapp pannu indha naal vara mudiyaathu", "SEND_WHATSAPP", {"contact": "boss", "message": "indha naal vara mudiyaathu"}),
    make("amit ko message karo ghar pahunch gaya", "SEND_WHATSAPP", {"contact": "amit", "message": "ghar pahunch gaya"}),
    make("didi ku message pannu saapitaen", "SEND_WHATSAPP", {"contact": "didi", "message": "saapitaen"}),
    make("nanban ku whatsapp karo party la vaa", "SEND_WHATSAPP", {"contact": "nanban", "message": "party la vaa"}),
]

# OPEN_APP (15)
cs_examples += [
    make("chrome open pannu", "OPEN_APP", {"app": "chrome"}),
    make("youtube open karo", "OPEN_APP", {"app": "youtube"}),
    make("settings kholo", "OPEN_APP", {"app": "settings"}),
    make("camera open pannu", "OPEN_APP", {"app": "camera"}),
    make("maps open karo", "OPEN_APP", {"app": "maps"}),
    make("whatsapp open pannu", "OPEN_APP", {"app": "whatsapp"}),
    make("spotify open karo", "OPEN_APP", {"app": "spotify"}),
    make("youtube thira", "OPEN_APP", {"app": "youtube"}),
    make("chrome kholo bhai", "OPEN_APP", {"app": "chrome"}),
    make("settings open pannu", "OPEN_APP", {"app": "settings"}),
    make("maps kholo", "OPEN_APP", {"app": "maps"}),
    make("camera kholo yaar", "OPEN_APP", {"app": "camera"}),
    make("whatsapp kholo", "OPEN_APP", {"app": "whatsapp"}),
    make("spotify open pannu", "OPEN_APP", {"app": "spotify"}),
    make("youtube la poo", "OPEN_APP", {"app": "youtube"}),
]

# NAVIGATE (10)
cs_examples += [
    make("airport navigate pannu", "NAVIGATE", {"destination": "airport"}),
    make("ghar ka rasta dikha", "NAVIGATE", {"destination": "ghar"}),
    make("hospital poga vazhi kaatu", "NAVIGATE", {"destination": "hospital"}),
    make("railway station jaana hai", "NAVIGATE", {"destination": "railway station"}),
    make("petrol pump navigate karo", "NAVIGATE", {"destination": "petrol pump"}),
    make("mall poga vazhi kaatu", "NAVIGATE", {"destination": "mall"}),
    make("office le poo", "NAVIGATE", {"destination": "office"}),
    make("bus stand ka rasta", "NAVIGATE", {"destination": "bus stand"}),
    make("airport kaise jayen", "NAVIGATE", {"destination": "airport"}),
    make("ATM poga vazhi", "NAVIGATE", {"destination": "ATM"}),
]

# WIFI/BLUETOOTH/HOME/BACK/LOCK/SCREENSHOT mixed (38)
cs_examples += [
    make("wifi on kar do", "TOGGLE_WIFI", {"state": "on"}),
    make("wifi off pannu", "TOGGLE_WIFI", {"state": "off"}),
    make("wifi on pannu", "TOGGLE_WIFI", {"state": "on"}),
    make("wifi band karo", "TOGGLE_WIFI", {"state": "off"}),
    make("net on kar do", "TOGGLE_WIFI", {"state": "on"}),
    make("internet off pannu", "TOGGLE_WIFI", {"state": "off"}),
    make("bluetooth on pannu", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("bluetooth off karo", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("bluetooth on kar do", "TOGGLE_BLUETOOTH", {"state": "on"}),
    make("bluetooth band pannu", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("ghar jaao", "GO_HOME"),
    make("home poo", "GO_HOME"),
    make("home screen pe jao", "GO_HOME"),
    make("home la poo", "GO_HOME"),
    make("wapas jaao", "GO_BACK"),
    make("back pannu", "GO_BACK"),
    make("pichi poo", "GO_BACK"),
    make("wapas jao yaar", "GO_BACK"),
    make("phone lock pannu", "LOCK_SCREEN"),
    make("screen lock karo", "LOCK_SCREEN"),
    make("phone lock kar do", "LOCK_SCREEN"),
    make("phone poottu", "LOCK_SCREEN"),
    make("screenshot lo", "TAKE_SCREENSHOT"),
    make("screenshot edu", "TAKE_SCREENSHOT"),
    make("screen capture pannu", "TAKE_SCREENSHOT"),
    make("screenshot le lo", "TAKE_SCREENSHOT"),
    make("torch on pannu da", "FLASHLIGHT_ON"),
    make("torch off pannu da", "FLASHLIGHT_OFF"),
    make("volume up pannu", "VOLUME_UP"),
    make("volume down pannu", "VOLUME_DOWN"),
    make("amma ko call karo", "CALL_CONTACT", {"contact": "amma"}),
    make("youtube open kar do", "OPEN_APP", {"app": "youtube"}),
    make("settings mela poo", "OPEN_APP", {"app": "settings"}),
    make("maps la poo airport", "NAVIGATE", {"destination": "airport"}),
    make("home button dabao", "GO_HOME"),
    make("back button dabao", "GO_BACK"),
    make("wifi chalú karo", "TOGGLE_WIFI", {"state": "on"}),
    make("bluetooth chalú karo", "TOGGLE_BLUETOOTH", {"state": "on"}),
]

# Whisper mishearing code-switched (10)
cs_examples += [
    make("torch on pan new", "FLASHLIGHT_ON"),
    make("amma cool pannu", "CALL_CONTACT", {"contact": "amma"}),
    make("alarm way kal 7 money", "SET_ALARM", {"time": "07:00", "date": "tomorrow"}),
    make("wahts app raj party eye aaj", "SEND_WHATSAPP", {"contact": "raj", "message": "party aaj"}),
    make("you tube open pan new", "OPEN_APP", {"app": "youtube"}),
    make("screen shot low", "TAKE_SCREENSHOT"),
    make("blue to off pan new", "TOGGLE_BLUETOOTH", {"state": "off"}),
    make("we fee on car do", "TOGGLE_WIFI", {"state": "on"}),
    make("volume bad how pan new", "VOLUME_UP"),
    make("wap pas jao", "GO_BACK"),
]

examples += cs_examples
print(f"Code-switched added. Total so far: {len(examples)}")

# ─────────────────────────────────────────────
# UNKNOWN — 100 examples
# ─────────────────────────────────────────────

unknown_examples = []
unknown_inputs = [
    "what is the weather today",
    "tell me a joke",
    "aaj kaisa din hai",
    "enna nadakudhu",
    "random blah blah xyz",
    "who is the prime minister",
    "what time is it",
    "who won the cricket match",
    "tell me a story",
    "what is 2 plus 2",
    "aaj ka mausam kaisa hai",
    "recipe batao biryani ki",
    "sing a song for me",
    "can you dance",
    "what is the meaning of life",
    "how are you",
    "kaisa chal raha hai",
    "epdi irukinga",
    "good morning",
    "namaste",
    "vanakkam",
    "hello how are you doing today",
    "tell me about history of india",
    "what is the capital of france",
    "latest news kya hai",
    "stock market kaisa hai",
    "cricket score kya hai",
    "enna time achu",
    "indha movie pathi sollu",
    "biryani recipe sollu",
    "poem padhu",
    "joke sollu",
    "how to cook rice",
    "what should i eat today",
    "aaj kya khana chahiye",
    "indha pudichirichu",
    "random random random",
    "asdfghjkl",
    "zxcvbnm",
    "1234567890",
    "delete all files",
    "format the phone",
    "hack my account",
    "send money to stranger",
    "what is artificial intelligence",
    "explain machine learning",
    "who is elon musk",
    "what is bitcoin",
    "IPL score kya hai",
    "cricket match ka result",
    "tell me something interesting",
    "what day is today",
    "aaj kaunsa din hai",
    "indha varam enna naal",
    "capital of america",
    "sun kya hota hai",
    "moon ke baare mein batao",
    "space ke baare mein batao",
    "give me a compliment",
    "am i beautiful",
    "how old are you",
    "where do you live",
    "are you human",
    "do you have feelings",
    "can you think",
    "apna naam batao",
    "unna peru enna",
    "what is your name",
    "what can you do",
    "help me",
    "I need assistance",
    "is it going to rain",
    "should i carry umbrella",
    "aaj baarish hogi kya",
    "mazhai varumo",
    "temperature kya hai bahar",
    "velicham epdi iruku",
    "kal holidays hai kya",
    "naalai holiday aa",
    "office band hai kya",
    "school open hai kya",
    "exam kab hai",
    "enna padikanum",
    "how to study better",
    "tips for sleeping well",
    "nalla tookam varathu",
    "neend nahi aa rahi",
    "stress kam kaise kare",
    "how to be happy",
    "life advice do",
    "motivate me",
    "i am sad",
    "mujhe bura lag raha hai",
    "enakku kodi iruku",
    "i am bored",
    "kuch karo entertainment ke liye",
    "mela kelvi ketkade",
    "just talking",
    "nothing special",
    "hi there",
    "bye bye",
    "thank you",
    "ok fine",
    "sure why not",
]

for text in unknown_inputs:
    unknown_examples.append(make(text, "UNKNOWN"))

examples += unknown_examples
print(f"UNKNOWN added. Total so far: {len(examples)}")

# ─────────────────────────────────────────────
# Shuffle and write
# ─────────────────────────────────────────────
random.seed(42)
random.shuffle(examples)

OUTPUT_FILE = "vanimitra_train.jsonl"
valid_count = 0
error_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in examples:
        try:
            line = json.dumps(ex, ensure_ascii=False)
            # Validate it's parseable
            _ = json.loads(line)
            f.write(line + "\n")
            valid_count += 1
        except Exception as e:
            print(f"ERROR on example: {e}")
            error_count += 1

print(f"\n{'='*50}")
print(f"✅ Written: {valid_count} examples")
print(f"❌ Errors:  {error_count}")
print(f"📄 Output:  {OUTPUT_FILE}")
print(f"{'='*50}")

# Distribution report
from collections import Counter
intents = []
langs = {"hindi": 0, "tamil": 0, "code_switched": 0, "unknown": 0}
with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        assistant_content = json.loads(obj["messages"][2]["content"])
        intents.append(assistant_content["intent"])

print("\nIntent distribution:")
for intent, count in sorted(Counter(intents).items()):
    print(f"  {intent:25s}: {count}")
