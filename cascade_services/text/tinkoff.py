from mutagen.mp3 import MP3

from tinkoff.cloud.stt.v1 import stt_pb2_grpc, stt_pb2
import grpc
import copy
from time import time
import json
import base64
import hmac

from cascade import CascadeBlock, CascadeElement
from collections import OrderedDict


def generate_jwt(api_key, secret_key, payload, expiration_time=6000):
    header = {
        "alg": "HS256",
        "typ": "JWT",
        "kid": api_key
    }
    payload_copy = copy.deepcopy(payload)
    current_timestamp = int(time())
    payload_copy["exp"] = current_timestamp + expiration_time

    payload_bytes = json.dumps(payload_copy, separators=(',', ':')).encode("utf-8")
    header_bytes = json.dumps(header, separators=(',', ':')).encode("utf-8")

    data = (base64.urlsafe_b64encode(header_bytes).strip(b'=') + b"." +
            base64.urlsafe_b64encode(payload_bytes).strip(b'='))

    signature = hmac.new(base64.urlsafe_b64decode(secret_key), msg=data, digestmod="sha256")
    jwt = data + b"." + base64.urlsafe_b64encode(signature.digest()).strip(b'=')
    return jwt.decode("utf-8")


def authorization_metadata(api_key, secret_key, scope, expiration_time):
    auth_payload = {
        "iss": "test_issuer",
        "sub": "test_user",
        "aud": scope
    }

    metadata = [
        ("authorization", "Bearer " + generate_jwt(api_key, secret_key, auth_payload, expiration_time=6000))
    ]
    return list(metadata)


def build_request(path: str, max_alternatives: int, do_not_perform_vad: bool, profanity_filter: bool):
    mp3_file = MP3(path)
    num_ch = int(mp3_file.info.channels)
    sr_audio = int(mp3_file.info.sample_rate)
    request = stt_pb2.RecognizeRequest()
    with open(path, "rb") as f:
        request.audio.content = f.read()

    request.config.encoding = stt_pb2.AudioEncoding.MPEG_AUDIO
    request.config.sample_rate_hertz = sr_audio
    request.config.num_channels = num_ch  # количество каналов в записи

    request.config.max_alternatives = max_alternatives  # включение альтернативных распознаваний
    request.config.do_not_perform_vad = do_not_perform_vad  # отключение режима диалога
    request.config.profanity_filter = profanity_filter  # фильтр ненормативной лексики
    return request


def response2result(response):
    tinkoff_res = ''
    for result in response.results:
        if int(result.channel) == int(0):
            ch = '-  '
        elif int(result.channel) == int(1):
            ch = '-- '

        for alternative in result.alternatives:
            tinkoff_res += '\n' + ch + alternative.transcript

    return tinkoff_res


class TinkoffSTT(CascadeBlock):
    """
    max_alternatives - включение альтернативных распознаваний
    do_not_perform_vad - отключение режима диалога
    profanity_filter - фильтр ненормативной лексики
    """
    def __init__(
            self, api_key: str, secret_key: str, max_alternatives: int = 3, do_not_perform_vad: bool = True,
            profanity_filter: bool = True, expiration_time: int = int(6e4), endpoint: str = 'stt.tinkoff.ru:443'
    ):
        stub = stt_pb2_grpc.SpeechToTextStub(grpc.secure_channel(endpoint, grpc.ssl_channel_credentials()))
        metadata = authorization_metadata(api_key, secret_key, "tinkoff.cloud.stt", expiration_time)

        self.response = CascadeElement(lambda path: stub.Recognize(build_request(
            path, max_alternatives, do_not_perform_vad, profanity_filter
        ), metadata=metadata), name="Build Request")

        self.make_result = CascadeElement(response2result, name="Make Result")

        self.adjacency_map = OrderedDict([
            (self.response, ['ITER']),
            (self.make_result, [self.response])
        ])

        super(TinkoffSTT, self).__init__(self.adjacency_map)
