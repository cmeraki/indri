import numpy as np
from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from fastapi import File, UploadFile

class Speakers(Enum):
    """
    Display name for the speakers on the front end
    """
    SPEAKER_1 = '[spkr_youtube_webds_en_historyofindia]'
    SPEAKER_2 = '[spkr_youtube_webds_en_mkbhd]'
    SPEAKER_3 = '[spkr_youtube_webds_en_secondhandstories]'
    SPEAKER_4 = '[spkr_youtube_webds_en_storiesofmahabharatha]'
    SPEAKER_5 = '[spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_6 = '[spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_7 = '[spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_8 = '[spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_9 = '[spkr_youtube_webds_en_derekperkins]'
    SPEAKER_10 = '[spkr_youtube_webds_en_mukesh]'
    SPEAKER_11 = '[spkr_youtube_webds_en_attenborough]'
    SPEAKER_12 = '[spkr_youtube_webds_hi_warikoo]'
    SPEAKER_13 = '[spkr_youtube_webds_hi_pmmodi]'

class MergedSpeakers(Enum):
    SPEAKER_1 = '[spkr_youtube_webds_en_historyofindia]'
    SPEAKER_2 = '[spkr_youtube_webds_en_mkbhd]'
    SPEAKER_3 = '[spkr_youtube_webds_en_secondhandstories]'
    SPEAKER_4 = '[spkr_youtube_webds_en_storiesofmahabharatha]'
    SPEAKER_5 = '[spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_6 = '[spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_7 = '[spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_8 = '[spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_9 = '[spkr_youtube_webds_en_derekperkins]'
    SPEAKER_10 = '[spkr_youtube_webds_en_mukesh]'
    SPEAKER_11 = '[spkr_youtube_webds_en_attenborough]'
    SPEAKER_12 = '[spkr_youtube_webds_hi_warikoo]'
    SPEAKER_13 = '[spkr_youtube_webds_hi_pmmodi]'

    SPEAKER_1_2 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_mkbhd]'
    SPEAKER_1_3 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_secondhandstories]'
    SPEAKER_1_4 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_storiesofmahabharatha]'
    SPEAKER_1_5 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_1_6 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_1_7 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_1_8 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_1_9 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_1_10 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_mukesh]'
    SPEAKER_1_11 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_attenborough]'
    SPEAKER_1_12 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_1_13 = '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_2_3 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_secondhandstories]'
    SPEAKER_2_4 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_storiesofmahabharatha]'
    SPEAKER_2_5 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_2_6 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_2_7 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_2_8 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_2_9 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_2_10 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_mukesh]'
    SPEAKER_2_11 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_attenborough]'
    SPEAKER_2_12 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_2_13 = '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_3_4 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_storiesofmahabharatha]'
    SPEAKER_3_5 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_3_6 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_3_7 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_3_8 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_3_9 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_3_10 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_mukesh]'
    SPEAKER_3_11 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_attenborough]'
    SPEAKER_3_12 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_3_13 = '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_4_5 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_4_6 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_4_7 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_4_8 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_4_9 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_4_10 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_en_mukesh]'
    SPEAKER_4_11 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_en_attenborough]'
    SPEAKER_4_12 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_4_13 = '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_5_6 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_5_7 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_5_8 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_5_9 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_5_10 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_en_mukesh]'
    SPEAKER_5_11 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_en_attenborough]'
    SPEAKER_5_12 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_5_13 = '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_6_7 = '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_6_8 = '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_6_9 = '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_6_10 = '[spkr_youtube_webds_hi_hindiaud:iobooks_spkr_youtube_webds_en_mukesh]'
    SPEAKER_6_11 = '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_en_attenborough]'
    SPEAKER_6_12 = '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_6_13 = '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_7_8 = '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_7_9 = '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_7_10 = '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_en_mukesh]'
    SPEAKER_7_11 = '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_en_attenborough]'
    SPEAKER_7_12 = '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_7_13 = '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_8_9 = '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_en_derekperkins]'
    SPEAKER_8_10 = '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_en_mukesh]'
    SPEAKER_8_11 = '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_en_attenborough]'
    SPEAKER_8_12 = '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_8_13 = '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_9_10 = '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_en_mukesh]'
    SPEAKER_9_11 = '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_en_attenborough]'
    SPEAKER_9_12 = '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_9_13 = '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_10_11 = '[spkr_youtube_webds_en_mukesh_spkr_youtube_webds_en_attenborough]'
    SPEAKER_10_12 = '[spkr_youtube_webds_en_mukesh_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_10_13 = '[spkr_youtube_webds_en_mukesh_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_11_12 = '[spkr_youtube_webds_en_attenborough_spkr_youtube_webds_hi_warikoo]'
    SPEAKER_11_13 = '[spkr_youtube_webds_en_attenborough_spkr_youtube_webds_hi_pmmodi]'
    SPEAKER_12_13 = '[spkr_youtube_webds_hi_warikoo_spkr_youtube_webds_hi_pmmodi]'

class TTSMetrics(BaseModel):
    time_to_first_token: List[float]
    time_to_last_token: List[float]
    time_to_encode_audio: Optional[float] = None
    time_to_decode_audio: float
    input_tokens: List[int]
    decoding_tokens: List[int]
    generate_end_to_end_time: float
    end_to_end_time: Optional[float] = None

class AudioContinuationMetrics(BaseModel):
    time_to_first_token: List[float]
    time_to_last_token: List[float]
    input_tokens: List[int]
    decoding_tokens: List[int]
    time_to_encode_audio: float
    time_to_decode_audio: Optional[float] = None
    generate_end_to_end_time: float
    end_to_end_time: Optional[float] = None

@dataclass
class AudioOutput:
    audio: np.ndarray
    sample_rate: int
    audio_metrics: Dict[str, Any]

class TTSSpeakersResponse(BaseModel):
    speakers: List[str]

class SpeakerTextRequest(BaseModel):
    speaker: Speakers

class SpeakerTextResponse(BaseModel):
    speaker_text: str

class AudioFeedbackRequest(BaseModel):
    id: str
    feedback: int

# Maps to speaker tags used in the model
SPEAKER_MAP = {
    Speakers.SPEAKER_1: {
        'id': '[spkr_youtube_webds_en_historyofindia]',
        'text': [
            "Today, we'll explore the fascinating journey of ancient India, from the Indus Valley Civilization to the Mauryan Empire.",
            "Let's delve into the rich tapestry of Indian history, examining the cultural and social developments that shaped the subcontinent.",
            "Welcome to another episode on the magnificent history of India, where we'll uncover forgotten stories and legendary tales."
        ]
    },
    Speakers.SPEAKER_2: {
        'id': '[spkr_youtube_webds_en_mkbhd]',
        'text': [
            "What's up guys, MKBHD here with another tech review that you've been waiting for.",
            "Let's talk about the future of technology and what these changes mean for all of us.",
            "After spending two weeks with this device, here are my detailed thoughts and impressions."
        ]
    },
    Speakers.SPEAKER_3: {
        'id': '[spkr_youtube_webds_en_secondhandstories]',
        'text': [
            "Every object has a story, and today we're uncovering the hidden history behind this remarkable find.",
            "Join me as we explore the fascinating world of vintage collectibles and their untold stories.",
            "What makes secondhand items special is not their value, but the memories they carry."
        ]
    },
    Speakers.SPEAKER_4: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha]',
        'text': [
            "Listen to the tale of Arjuna and Krishna, as they discuss dharma on the battlefield of Kurukshetra.",
            "Today we explore the wisdom of the Pandavas and the complex moral choices they faced.",
            "The Mahabharata teaches us about duty, honor, and the eternal struggle between right and wrong."
        ]
    },
    Speakers.SPEAKER_5: {
        'id': '[spkr_youtube_webds_hi_a2motivation]',
        'text': [
            "दोस्तों, आज हम बात करेंगे सफलता की असली कुंजी के बारे में।",
            "जीवन में सफल होने के लिए सबसे जरूरी है अपने लक्ष्य के प्रति समर्पण।",
            "याद रखिए, हर असफलता एक नई सीख लेकर आती है।"
        ]
    },
    Speakers.SPEAKER_6: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks]',
        'text': [
            "प्रेमचंद की इस अमर कहानी को सुनिए, जो आज भी उतनी ही प्रासंगिक है।",
            "आज हम पढ़ेंगे हिंदी साहित्य का एक अनमोल रत्न।",
            "इस कहानी में छिपा है जीवन का गहरा सत्य।"
        ]
    },
    Speakers.SPEAKER_7: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen]',
        'text': [
            "आज हम बनाएंगे एकदम परफेक्ट दाल मखनी, जो होटल जैसी टेस्टी बनेगी।",
            "नमस्ते दोस्तों, आज की रेसिपी है स्पेशल पनीर टिक्का मसाला।",
            "इस ट्रिक से आपकी सब्जी होटल जैसी बनेगी।"
        ]
    },
    Speakers.SPEAKER_8: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks]',
        'text': [
            "आज की कहानी है प्रेम और त्याग की एक अनूठी गाथा।",
            "हिंदी साहित्य की इस क्लासिक रचना को सुनिए मेरी आवाज में।",
            "कहानी शुरू करने से पहले आप सभी का स्वागत है।"
        ]
    },
    Speakers.SPEAKER_9: {
        'id': '[spkr_youtube_webds_en_derekperkins]',
        'text': [
            "Chapter One: The distant horizon stretched endlessly before us, as we embarked on our journey.",
            "In the stillness of the night, the ancient prophecy began to unfold.",
            "The characters in this story will take you on an unforgettable adventure."
        ]
    },
    Speakers.SPEAKER_10: {
        'id': '[spkr_youtube_webds_en_mukesh]',
        'text': [
            "Education is not just about degrees, it's about becoming a better human being.",
            "Success comes to those who are willing to push beyond their comfort zones.",
            "Today, let's discuss how to transform your dreams into achievable goals."
        ]
    },
    Speakers.SPEAKER_11: {
        'id': '[spkr_youtube_webds_en_attenborough]',
        'text': [
            "Here in the heart of the rainforest, a remarkable story of survival unfolds.",
            "These extraordinary creatures have adapted to some of the most extreme conditions on Earth.",
            "What we're witnessing is one of nature's most spectacular displays."
        ]
    },
    Speakers.SPEAKER_12: {
        'id': '[spkr_youtube_webds_hi_warikoo]',
        'text': [
            "दोस्तों, आज बात करेंगे कैरियर, पैसे और सफलता के बारे में।",
            "क्या आप जानते हैं कि फाइनेंशियल फ्रीडम का असली मतलब क्या है?",
            "मैंने अपनी जिंदगी में जो गलतियां की, आप उनसे सीख सकते हैं।"
        ]
    },
    Speakers.SPEAKER_13: {
        'id': '[spkr_youtube_webds_hi_pmmodi]',
        'text': [
            "मेरे प्यारे देशवासियों, आज हम एक नए भारत की ओर कदम बढ़ा रहे हैं।",
            "हमारा संकल्प है - एक भारत, श्रेष्ठ भारत।",
            "युवा शक्ति ही देश की सबसे बड़ी ताकत है।"
        ]
    },
    # TODO: This will be cleaned up once we have finalized the speakers
    MergedSpeakers.SPEAKER_1: {
        'id': '[spkr_youtube_webds_en_historyofindia]',
        'text': [
            "Today, we'll explore the fascinating journey of ancient India, from the Indus Valley Civilization to the Mauryan Empire.",
            "Let's delve into the rich tapestry of Indian history, examining the cultural and social developments that shaped the subcontinent.",
            "Welcome to another episode on the magnificent history of India, where we'll uncover forgotten stories and legendary tales."
        ]
    },
    MergedSpeakers.SPEAKER_2: {
        'id': '[spkr_youtube_webds_en_mkbhd]',
        'text': [
            "What's up guys, MKBHD here with another tech review that you've been waiting for.",
            "Let's talk about the future of technology and what these changes mean for all of us.",
            "After spending two weeks with this device, here are my detailed thoughts and impressions."
        ]
    },
    MergedSpeakers.SPEAKER_3: {
        'id': '[spkr_youtube_webds_en_secondhandstories]',
        'text': [
            "Every object has a story, and today we're uncovering the hidden history behind this remarkable find.",
            "Join me as we explore the fascinating world of vintage collectibles and their untold stories.",
            "What makes secondhand items special is not their value, but the memories they carry."
        ]
    },
    MergedSpeakers.SPEAKER_4: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha]',
        'text': [
            "Listen to the tale of Arjuna and Krishna, as they discuss dharma on the battlefield of Kurukshetra.",
            "Today we explore the wisdom of the Pandavas and the complex moral choices they faced.",
            "The Mahabharata teaches us about duty, honor, and the eternal struggle between right and wrong."
        ]
    },
    MergedSpeakers.SPEAKER_5: {
        'id': '[spkr_youtube_webds_hi_a2motivation]',
        'text': [
            "दोस्तों, आज हम बात करेंगे सफलता की असली कुंजी के बारे में।",
            "जीवन में सफल होने के लिए सबसे जरूरी है अपने लक्ष्य के प्रति समर्पण।",
            "याद रखिए, हर असफलता एक नई सीख लेकर आती है।"
        ]
    },
    MergedSpeakers.SPEAKER_6: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks]',
        'text': [
            "प्रेमचंद की इस अमर कहानी को सुनिए, जो आज भी उतनी ही प्रासंगिक है।",
            "आज हम पढ़ेंगे हिंदी साहित्य का एक अनमोल रत्न।",
            "इस कहानी में छिपा है जीवन का गहरा सत्य।"
        ]
    },
    MergedSpeakers.SPEAKER_7: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen]',
        'text': [
            "आज हम बनाएंगे एकदम परफेक्ट दाल मखनी, जो होटल जैसी टेस्टी बनेगी।",
            "नमस्ते दोस्तों, आज की रेसिपी है स्पेशल पनीर टिक्का मसाला।",
            "इस ट्रिक से आपकी सब्जी होटल जैसी बनेगी।"
        ]
    },
    MergedSpeakers.SPEAKER_8: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks]',
        'text': [
            "आज की कहानी है प्रेम और त्याग की एक अनूठी गाथा।",
            "हिंदी साहित्य की इस क्लासिक रचना को सुनिए मेरी आवाज में।",
            "कहानी शुरू करने से पहले आप सभी का स्वागत है।"
        ]
    },
    MergedSpeakers.SPEAKER_9: {
        'id': '[spkr_youtube_webds_en_derekperkins]',
        'text': [
            "Chapter One: The distant horizon stretched endlessly before us, as we embarked on our journey.",
            "In the stillness of the night, the ancient prophecy began to unfold.",
            "The characters in this story will take you on an unforgettable adventure."
        ]
    },
    MergedSpeakers.SPEAKER_10: {
        'id': '[spkr_youtube_webds_en_mukesh]',
        'text': [
            "Education is not just about degrees, it's about becoming a better human being.",
            "Success comes to those who are willing to push beyond their comfort zones.",
            "Today, let's discuss how to transform your dreams into achievable goals."
        ]
    },
    MergedSpeakers.SPEAKER_11: {
        'id': '[spkr_youtube_webds_en_attenborough]',
        'text': [
            "Here in the heart of the rainforest, a remarkable story of survival unfolds.",
            "These extraordinary creatures have adapted to some of the most extreme conditions on Earth.",
            "What we're witnessing is one of nature's most spectacular displays."
        ]
    },
    MergedSpeakers.SPEAKER_12: {
        'id': '[spkr_youtube_webds_hi_warikoo]',
        'text': [
            "दोस्तों, आज बात करेंगे कैरियर, पैसे और सफलता के बारे में।",
            "क्या आप जानते हैं कि फाइनेंशियल फ्रीडम का असली मतलब क्या है?",
            "मैंने अपनी जिंदगी में जो गलतियां की, आप उनसे सीख सकते हैं।"
        ]
    },
    MergedSpeakers.SPEAKER_13: {
        'id': '[spkr_youtube_webds_hi_pmmodi]',
        'text': [
            "मेरे प्यारे देशवासियों, आज हम एक नए भारत की ओर कदम बढ़ा रहे हैं।",
            "हमारा संकल्प है - एक भारत, श्रेष्ठ भारत।",
            "युवा शक्ति ही देश की सबसे बड़ी ताकत है।"
        ]
    },
    MergedSpeakers.SPEAKER_1_2: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_mkbhd]'
    },
    MergedSpeakers.SPEAKER_1_3: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_secondhandstories]'
    },
    MergedSpeakers.SPEAKER_1_4: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_storiesofmahabharatha]'
    },
    MergedSpeakers.SPEAKER_1_5: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_a2motivation]'
    },
    MergedSpeakers.SPEAKER_1_6: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_hindiaudiobooks]'
    },
    MergedSpeakers.SPEAKER_1_7: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_kabitaskitchen]'
    },
    MergedSpeakers.SPEAKER_1_8: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_1_9: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_1_10: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_1_11: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_1_12: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_1_13: {
        'id': '[spkr_youtube_webds_en_historyofindia_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_2_3: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_secondhandstories]'
    },
    MergedSpeakers.SPEAKER_2_4: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_storiesofmahabharatha]'
    },
    MergedSpeakers.SPEAKER_2_5: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_a2motivation]'
    },
    MergedSpeakers.SPEAKER_2_6: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_hindiaudiobooks]'
    },
    MergedSpeakers.SPEAKER_2_7: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_kabitaskitchen]'
    },
    MergedSpeakers.SPEAKER_2_8: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_2_9: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_2_10: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_2_11: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_2_12: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_2_13: {
        'id': '[spkr_youtube_webds_en_mkbhd_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_3_4: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_storiesofmahabharatha]'
    },
    MergedSpeakers.SPEAKER_3_5: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_a2motivation]'
    },
    MergedSpeakers.SPEAKER_3_6: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_hindiaudiobooks]'
    },
    MergedSpeakers.SPEAKER_3_7: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_kabitaskitchen]'
    },
    MergedSpeakers.SPEAKER_3_8: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_3_9: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_3_10: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_3_11: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_3_12: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_3_13: {
        'id': '[spkr_youtube_webds_en_secondhandstories_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_4_5: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_a2motivation]'
    },
    MergedSpeakers.SPEAKER_4_6: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_hindiaudiobooks]'
    },
    MergedSpeakers.SPEAKER_4_7: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_kabitaskitchen]'
    },
    MergedSpeakers.SPEAKER_4_8: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_4_9: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_4_10: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_4_11: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_4_12: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_4_13: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_5_6: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_hindiaudiobooks]'
    },
    MergedSpeakers.SPEAKER_5_7: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_kabitaskitchen]'
    },
    MergedSpeakers.SPEAKER_5_8: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_5_9: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_5_10: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_5_11: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_5_12: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_5_13: {
        'id': '[spkr_youtube_webds_hi_a2motivation_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_6_7: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_kabitaskitchen]'
    },
    MergedSpeakers.SPEAKER_6_8: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_6_9: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_6_10: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_6_11: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_6_12: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_6_13: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_7_8: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_hi_neelimaaudiobooks]'
    },
    MergedSpeakers.SPEAKER_7_9: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_7_10: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_7_11: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_7_12: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_7_13: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_8_9: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_en_derekperkins]'
    },
    MergedSpeakers.SPEAKER_8_10: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_8_11: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_8_12: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_8_13: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_9_10: {
        'id': '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_en_mukesh]'
    },
    MergedSpeakers.SPEAKER_9_11: {
        'id': '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_9_12: {
        'id': '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_9_13: {
        'id': '[spkr_youtube_webds_en_derekperkins_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_10_11: {
        'id': '[spkr_youtube_webds_en_mukesh_spkr_youtube_webds_en_attenborough]'
    },
    MergedSpeakers.SPEAKER_10_12: {
        'id': '[spkr_youtube_webds_en_mukesh_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_10_13: {
        'id': '[spkr_youtube_webds_en_mukesh_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_11_12: {
        'id': '[spkr_youtube_webds_en_attenborough_spkr_youtube_webds_hi_warikoo]'
    },
    MergedSpeakers.SPEAKER_11_13: {
        'id': '[spkr_youtube_webds_en_attenborough_spkr_youtube_webds_hi_pmmodi]'
    },
    MergedSpeakers.SPEAKER_12_13: {
        'id': '[spkr_youtube_webds_hi_warikoo_spkr_youtube_webds_hi_pmmodi]'
    }
}
