import numpy as np
from enum import Enum
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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


class SpeakerInfo(BaseModel):
    id: str
    text: List[str]
    compatible_speakers: List[Speakers]


SPEAKER_MAP: Dict[Speakers, SpeakerInfo] = {
    Speakers.SPEAKER_1: {
        'id': '[spkr_youtube_webds_en_historyofindia]',
        'text': [
            "Today, we'll explore the fascinating journey of ancient India, from the Indus Valley Civilization to the Mauryan Empire.",
            "Let's delve into the rich tapestry of Indian history, examining the cultural and social developments that shaped the subcontinent.",
            "Welcome to another episode on the magnificent history of India, where we'll uncover forgotten stories and legendary tales."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_2,
            Speakers.SPEAKER_3,
            Speakers.SPEAKER_4,
            Speakers.SPEAKER_5,
            Speakers.SPEAKER_6,
            Speakers.SPEAKER_7,
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_2: {
        'id': '[spkr_youtube_webds_en_mkbhd]',
        'text': [
            "What's up guys, MKBHD here with another tech review that you've been waiting for.",
            "Let's talk about the future of technology and what these changes mean for all of us.",
            "After spending two weeks with this device, here are my detailed thoughts and impressions."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_2,
            Speakers.SPEAKER_3,
            Speakers.SPEAKER_4,
            Speakers.SPEAKER_5,
            Speakers.SPEAKER_6,
            Speakers.SPEAKER_7,
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_3: {
        'id': '[spkr_youtube_webds_en_secondhandstories]',
        'text': [
            "Every object has a story, and today we're uncovering the hidden history behind this remarkable find.",
            "Join me as we explore the fascinating world of vintage collectibles and their untold stories.",
            "What makes secondhand items special is not their value, but the memories they carry."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_4,
            Speakers.SPEAKER_5,
            Speakers.SPEAKER_6,
            Speakers.SPEAKER_7,
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_4: {
        'id': '[spkr_youtube_webds_en_storiesofmahabharatha]',
        'text': [
            "Listen to the tale of Arjuna and Krishna, as they discuss dharma on the battlefield of Kurukshetra.",
            "Today we explore the wisdom of the Pandavas and the complex moral choices they faced.",
            "The Mahabharata teaches us about duty, honor, and the eternal struggle between right and wrong."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_5,
            Speakers.SPEAKER_6,
            Speakers.SPEAKER_7,
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_5: {
        'id': '[spkr_youtube_webds_hi_a2motivation]',
        'text': [
            "दोस्तों, आज हम बात करेंगे सफलता की असली कुंजी के बारे में।",
            "जीवन में सफल होने के लिए सबसे जरूरी है अपने लक्ष्य के प्रति समर्पण।",
            "याद रखिए, हर असफलता एक नई सीख लेकर आती है।"
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_6,
            Speakers.SPEAKER_7,
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_6: {
        'id': '[spkr_youtube_webds_hi_hindiaudiobooks]',
        'text': [
            "प्रेमचंद की इस अमर कहानी को सुनिए, जो आज भी उतनी ही प्रासंगिक है।",
            "आज हम पढ़ेंगे हिंदी साहित्य का एक अनमोल रत्न।",
            "इस कहानी में छिपा है जीवन का गहरा सत्य।"
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_7,
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_7: {
        'id': '[spkr_youtube_webds_hi_kabitaskitchen]',
        'text': [
            "आज हम बनाएंगे एकदम परफेक्ट दाल मखनी, जो होटल जैसी टेस्टी बनेगी।",
            "नमस्ते दोस्तों, आज की रेसिपी है स्पेशल पनीर टिक्का मसाला।",
            "इस ट्रिक से आपकी सब्जी होटल जैसी बनेगी।"
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_8,
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_8: {
        'id': '[spkr_youtube_webds_hi_neelimaaudiobooks]',
        'text': [
            "आज की कहानी है प्रेम और त्याग की एक अनूठी गाथा।",
            "हिंदी साहित्य की इस क्लासिक रचना को सुनिए मेरी आवाज में।",
            "कहानी शुरू करने से पहले आप सभी का स्वागत है।"
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_9,
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_9: {
        'id': '[spkr_youtube_webds_en_derekperkins]',
        'text': [
            "Chapter One: The distant horizon stretched endlessly before us, as we embarked on our journey.",
            "In the stillness of the night, the ancient prophecy began to unfold.",
            "The characters in this story will take you on an unforgettable adventure."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_10,
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_10: {
        'id': '[spkr_youtube_webds_en_mukesh]',
        'text': [
            "Education is not just about degrees, it's about becoming a better human being.",
            "Success comes to those who are willing to push beyond their comfort zones.",
            "Today, let's discuss how to transform your dreams into achievable goals."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_11,
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_11: {
        'id': '[spkr_youtube_webds_en_attenborough]',
        'text': [
            "Here in the heart of the rainforest, a remarkable story of survival unfolds.",
            "These extraordinary creatures have adapted to some of the most extreme conditions on Earth.",
            "What we're witnessing is one of nature's most spectacular displays."
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_12,
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_12: {
        'id': '[spkr_youtube_webds_hi_warikoo]',
        'text': [
            "दोस्तों, आज बात करेंगे कैरियर, पैसे और सफलता के बारे में।",
            "क्या आप जानते हैं कि फाइनेंशियल फ्रीडम का असली मतलब क्या है?",
            "मैंने अपनी जिंदगी में जो गलतियां की, आप उनसे सीख सकते हैं।"
        ],
        'compatible_speakers': [
            Speakers.SPEAKER_13,
        ]
    },
    Speakers.SPEAKER_13: {
        'id': '[spkr_youtube_webds_hi_pmmodi]',
        'text': [
            "मेरे प्यारे देशवासियों, आज हम एक नए भारत की ओर कदम बढ़ा रहे हैं।",
            "हमारा संकल्प है - एक भारत, श्रेष्ठ भारत।",
            "युवा शक्ति ही देश की सबसे बड़ी ताकत है।"
        ],
        'compatible_speakers': []
    },
}


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

