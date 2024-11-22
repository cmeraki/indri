import numpy as np
from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


class Speakers(Enum):
    """
    Display name for the speakers on the front end
    """

    SPEAKER_1 = '🇬🇧 👨 book reader'  #63
    SPEAKER_2 = '🇺🇸 👨 influencer'          #67
    SPEAKER_3 = '🇮🇳 👨 book reader'  #68
    SPEAKER_4 = '🇮🇳 👨 book reader'  #69
    SPEAKER_5 = '🇮🇳 👨 motivational speaker'   #70
    SPEAKER_6 = '🇮🇳 👨 book reader heavy'  #62
    SPEAKER_7 = '🇮🇳 👩 recipe reciter'  #53
    SPEAKER_8 = '🇮🇳 👩 book reader'  #60
    SPEAKER_9 = '🇺🇸 👨 book reader'   #74
    SPEAKER_10 = '🇮🇳 👨 entrepreneur'        #75
    SPEAKER_11 = '🇬🇧 👨 nature lover'  #76
    SPEAKER_12 = '🇮🇳 👨 influencer'       #77
    SPEAKER_13 = '🇮🇳 👨 politician'        #66

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

class TTSRequest(BaseModel):
    text: str
    speaker: Speakers

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
        'id': '[spkr_63]',
        'text': [
            "Today, we'll explore the fascinating journey of ancient India, from the Indus Valley Civilization to the Mauryan Empire.",
            "Let's delve into the rich tapestry of Indian history, examining the cultural and social developments that shaped the subcontinent.",
            "Welcome to another episode on the magnificent history of India, where we'll uncover forgotten stories and legendary tales."
        ]
    },
    Speakers.SPEAKER_2: {
        'id': '[spkr_67]',
        'text': [
            "What's up guys, MKBHD here with another tech review that you've been waiting for.",
            "Let's talk about the future of technology and what these changes mean for all of us.",
            "After spending two weeks with this device, here are my detailed thoughts and impressions."
        ]
    },
    Speakers.SPEAKER_3: {
        'id': '[spkr_68]',
        'text': [
            "Every object has a story, and today we're uncovering the hidden history behind this remarkable find.",
            "Join me as we explore the fascinating world of vintage collectibles and their untold stories.",
            "What makes secondhand items special is not their value, but the memories they carry."
        ]
    },
    Speakers.SPEAKER_4: {
        'id': '[spkr_69]',
        'text': [
            "Listen to the tale of Arjuna and Krishna, as they discuss dharma on the battlefield of Kurukshetra.",
            "Today we explore the wisdom of the Pandavas and the complex moral choices they faced.",
            "The Mahabharata teaches us about duty, honor, and the eternal struggle between right and wrong."
        ]
    },
    Speakers.SPEAKER_5: {
        'id': '[spkr_70]',
        'text': [
            "दोस्तों, आज हम बात करेंगे सफलता की असली कुंजी के बारे में।",
            "जीवन में सफल होने के लिए सबसे जरूरी है अपने लक्ष्य के प्रति समर्पण।",
            "याद रखिए, हर असफलता एक नई सीख लेकर आती है।"
        ]
    },
    Speakers.SPEAKER_6: {
        'id': '[spkr_62]',
        'text': [
            "प्रेमचंद की इस अमर कहानी को सुनिए, जो आज भी उतनी ही प्रासंगिक है।",
            "आज हम पढ़ेंगे हिंदी साहित्य का एक अनमोल रत्न।",
            "इस कहानी में छिपा है जीवन का गहरा सत्य।"
        ]
    },
    Speakers.SPEAKER_7: {
        'id': '[spkr_53]',
        'text': [
            "आज हम बनाएंगे एकदम परफेक्ट दाल मखनी, जो होटल जैसी टेस्टी बनेगी।",
            "नमस्ते दोस्तों, आज की रेसिपी है स्पेशल पनीर टिक्का मसाला।",
            "इस ट्रिक से आपकी सब्जी होटल जैसी बनेगी।"
        ]
    },
    Speakers.SPEAKER_8: {
        'id': '[spkr_60]',
        'text': [
            "आज की कहानी है प्रेम और त्याग की एक अनूठी गाथा।",
            "हिंदी साहित्य की इस क्लासिक रचना को सुनिए मेरी आवाज में।",
            "कहानी शुरू करने से पहले आप सभी का स्वागत है।"
        ]
    },
    Speakers.SPEAKER_9: {
        'id': '[spkr_74]',
        'text': [
            "Chapter One: The distant horizon stretched endlessly before us, as we embarked on our journey.",
            "In the stillness of the night, the ancient prophecy began to unfold.",
            "The characters in this story will take you on an unforgettable adventure."
        ]
    },
    Speakers.SPEAKER_10: {
        'id': '[spkr_75]',
        'text': [
            "Education is not just about degrees, it's about becoming a better human being.",
            "Success comes to those who are willing to push beyond their comfort zones.",
            "Today, let's discuss how to transform your dreams into achievable goals."
        ]
    },
    Speakers.SPEAKER_11: {
        'id': '[spkr_76]',
        'text': [
            "Here in the heart of the rainforest, a remarkable story of survival unfolds.",
            "These extraordinary creatures have adapted to some of the most extreme conditions on Earth.",
            "What we're witnessing is one of nature's most spectacular displays."
        ]
    },
    Speakers.SPEAKER_12: {
        'id': '[spkr_77]',
        'text': [
            "दोस्तों, आज बात करेंगे कैरियर, पैसे और सफलता के बारे में।",
            "क्या आप जानते हैं कि फाइनेंशियल फ्रीडम का असली मतलब क्या है?",
            "मैंने अपनी जिंदगी में जो गलतियां की, आप उनसे सीख सकते हैं।"
        ]
    },
    Speakers.SPEAKER_13: {
        'id': '[spkr_66]',
        'text': [
            "मेरे प्यारे देशवासियों, आज हम एक नए भारत की ओर कदम बढ़ा रहे हैं।",
            "हमारा संकल्प है - एक भारत, श्रेष्ठ भारत।",
            "युवा शक्ति ही देश की सबसे बड़ी ताकत है।"
        ]
    }
}
