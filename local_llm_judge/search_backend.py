import functools
from typing import Any, Optional
from elasticsearch import Elasticsearch

import pandas as pd
import pickle
import requests
import logging

import json

from google.cloud import secretmanager
from google.protobuf import wrappers_pb2

from liaison_client.generated_types.dahlialabs.liaison.v1beta1 import product_pb2
from liaison_client import Client, Config
from liaison_client.helpers import replay_chat
from liaison_client.exceptions import RpcError

from local_llm_judge.es_metadata import es_metadata, es_stag, es_prod

## ALL SEARCH BAKEND CODE EXPECTS TO BE RUN
## AFTER source ./connect.sh and its various port fwds


logger = logging.getLogger(__name__)


secret_manager_client = secretmanager.SecretManagerServiceClient()


def liaison_key(project="dahlia-infra-stag"):
    key = secret_manager_client.access_secret_version(
        name=f"projects/{project}/secrets/liaison-global-apiKey/versions/latest"
    ).payload.data.decode("utf-8")
    return key


inference_cache_path = "data/inference_cache.pkl"
inference_cache = {}
try:
    with open(inference_cache_path, 'rb') as f:
        inference_cache = pickle.load(f)
        logger.info(f"Loaded {len(inference_cache)} entries from inference cache")
except FileNotFoundError:
    inference_cache = {}
    logger.info("No inference cache found")


def _inference(inference_uri, text: str):
    response = requests.post(inference_uri,
                             json={'texts': [text]})
    textVector = response.json()['textVectors'][0]
    inference_cache[text] = textVector
    with open(inference_cache_path, 'wb') as f:
        pickle.dump(inference_cache, f)
    return textVector


def local_liaison():
    liaison_config = Config(
        host='127.0.0.1:8880',
        api_token=None,
        default_timeout_ms=30000,
        use_tls=False
    )
    return Client(config=liaison_config)


@functools.lru_cache()
def stag_liaison():
    uri = 'liaison.stag.dahlialabs.dev:443'
    token = liaison_key(project="dahlia-infra-stag")
    liaison_config = Config(
        host=uri,
        api_token=token,
        default_timeout_ms=30000,
        use_tls=True
    )
    return Client(config=liaison_config)


@functools.lru_cache()
def prod_liaison():
    uri = 'liaison.dahlialabs.dev:443'
    token = liaison_key(project="dahlia-infra-prod")
    liaison_config = Config(
        host=uri,
        api_token=token,
        default_timeout_ms=30000,
        use_tls=True
    )
    return Client(config=liaison_config)


def search_settings_proto_to_dict(search_settings_proto):
    def _to_list(proto_list):
        return [str(item) for item in proto_list]

    def _filter_to_dict(filters):
        return {
            'brand': _to_list(filters.brand),
            'color': _to_list(filters.color),
            'department': _to_list(filters.department),
            'material': _to_list(filters.material),
            'tag': _to_list(filters.tag),
            'category': _to_list(filters.category),
            'option_id': _to_list(filters.option_id),
        }

    search_settings = {
        "search_string": search_settings_proto.search_string,
        "minimum_price": search_settings_proto.minimum_price,
        "maximum_price": search_settings_proto.maximum_price,
        "lexical_search": search_settings_proto.lexical_search,
        "image_search": search_settings_proto.image_search,
    }
    positive_filters = search_settings_proto.positive_filters
    if positive_filters:
        search_settings["positive_filters"] = _filter_to_dict(positive_filters)
    negative_filters = search_settings_proto.negative_filters
    if negative_filters:
        search_settings["negative_filters"] = _filter_to_dict(negative_filters)
    return search_settings


def liaison_prods_from_msgs(liaison_client, es_client, user_messages, user_id,
                            N=10):
    agent_resp = None
    last_error = None
    for i in range(3):
        try:
            agent_resp = replay_chat(liaison_client, user_messages, user_id)
            break
        except RpcError as e:
            logger.warning(f"Error replaying chat: {e}. Retrying...")
            last_error = e
            continue
    if not agent_resp:
        raise ValueError("Failed to replay chat") from last_error
    chat_id = agent_resp.id
    listProductsReq = product_pb2.ListProductCardsRequest(
        page_size=10,
        chat_search_context=product_pb2.ListProductCardsSearchContext(
            chat_id=wrappers_pb2.StringValue(value=chat_id)  # pylint: disable=no-member
        )
    )
    listProductsReq.chat_search_context.search_settings.MergeFrom(agent_resp.search_settings)
    search_settings_dict = search_settings_proto_to_dict(agent_resp.search_settings)
    logger.debug(f"Search settings: {search_settings_dict}")
    result = liaison_client.product.list_product_cards(request=listProductsReq)
    ranked_options = []
    for product in result.products:
        product_name = product.name
        brand_name = product.brand_data.name
        description = product.description
        for option in product.options:
            if not option.provenance.source:
                break
            main_image_url = None
            for image in option.images:
                if image.is_main:
                    main_image_url = image.src
                    break
            ranked_options.append({
                "product_name": product_name,
                "brand_name": brand_name,
                "option_id": option.id,
                "main_image": main_image_url,
                'main_image_path': fetch_and_resize(main_image_url, option.id),
                "product_description": description
            })
            if len(ranked_options) >= N:
                break
    df = pd.DataFrame(ranked_options)
    extra_metadata = es_metadata(es_client, df['option_id'].tolist())
    df = df.merge(extra_metadata[['option_id', 'category']], on='option_id')
    df.rename(columns={'option_id': 'id', 'product_name': 'name', 'product_description': 'description'}, inplace=True)
    return df, search_settings_dict


class SearchBackend:
    def __init__(self, liaison_client, es_client, user_ids, name):
        self.liaison_client = liaison_client
        self.es_client = es_client
        self.user_ids = user_ids
        self.name = name

    def search(self, user_messages, dept):
        user_id = self.user_ids[dept]
        products, search_settings = liaison_prods_from_msgs(self.liaison_client, self.es_client, user_messages, user_id)
        products['backend'] = self.name
        return products, search_settings


class MemoizedSearchBackend:
    def __init__(self, backend):
        self.backend = backend
        self.cache = {}

    @property
    def name(self):
        return "memoized_" + self.backend.name

    def search(self, user_messages, dept):
        if (user_messages, dept) in self.cache:
            return self.cache[(user_messages, dept)]
        return self.backend.search(user_messages, dept)


class CachedSearchBackend:
    def __init__(self, backend):
        self.backend = backend
        self.cache_file = f"{backend.name}_search_results.pkl"
        self.cache = {}
        try:
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} entries from search backend cache")
        except FileNotFoundError:
            pass

    @property
    def name(self):
        return "cached_" + self.backend.name

    def search(self, user_messages, dept):
        if (user_messages, dept) in self.cache:
            return self.cache[(user_messages, dept)]
        result = self.backend.search(user_messages, dept)
        with open(self.cache_file, 'wb') as f:
            self.cache[(user_messages, dept)] = result
            pickle.dump(self.cache, f)
        if len(result[0]) > 10:
            return result[0].head(10), result[1]
        return result


stag_user_ids = {
    "w": "46467832-d7e9-43b9-9946-e0c00a1f7a76",
    "m": "c9d43820-3c20-415a-8a43-059fe0385716"
}

prod_user_ids = {
    "w": "9c61d6a6-a5fa-4cb4-bde5-485fc3231ae3",
    "m": "aa58aa2a-6902-4c86-afdf-62fafdb4b9ac"
}


stag = SearchBackend(stag_liaison(), es_stag, stag_user_ids, 'stag')
prod = SearchBackend(prod_liaison(), es_prod, prod_user_ids, 'prod')

stag_memoized = MemoizedSearchBackend(stag)
prod_memoized = MemoizedSearchBackend(prod)

stag_cached = CachedSearchBackend(stag)
prod_cached = CachedSearchBackend(prod)

# Local uses staging data
local = SearchBackend(local_liaison(), es_stag, stag_user_ids, 'local')


def search_template_params(
    keywords: str,
    brands: Optional[list[str]] = None,
    departments: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
    min_price: int = 0,
    max_price: int = 1000000,
    sizes: Optional[list[str]] = None,
    image_vector: Optional[list[float]] = None,
    brand_ages: Optional[list[str]] = None,
    brand_aesthetics: Optional[list[str]] = None,
    brand_occasions: Optional[list[str]] = None,
    brand_prices: Optional[list[str]] = None,
) -> dict[str, Any]:  # pylint: disable=too-many-arguments
    """Returns a dictionary of search template parameters."""

    def _filter_params(values: list[str]) -> list[dict[str, Any]]:
        return [{"value": value, "last": i == len(values) - 1} for i, value in enumerate(values)]

    params: dict[str, Any] = {
        "keywords": keywords,
    }
    if brands:
        params["brands"] = _filter_params(brands)
    if brand_ages:
        params["brand_ages"] = _filter_params(brand_ages)
    if brand_aesthetics:
        params["brand_aesthetics"] = _filter_params(brand_aesthetics)
    if brand_occasions:
        params["brand_occasions"] = _filter_params(brand_occasions)
    if brand_prices:
        params["brand_prices"] = _filter_params(brand_prices)
    if departments:
        params["departments"] = _filter_params(departments)
    if categories:
        params["categories"] = _filter_params(categories)
    if sizes:
        params["sizes"] = _filter_params(sizes)
    if image_vector:
        params["image_vector"] = json.dumps(image_vector)
    if min_price != max_price:
        params["min_price"] = str(min_price)
        params["max_price"] = str(max_price)
    else:
        params["min_price"] = str(0)
        params["max_price"] = str(int(2**31 - 1))
    return params


def option_details(liaison_client, option_id):
    try:
        option_details_req = product_pb2.GetOptionDetailsRequest(option_id=option_id)
        option_details_resp = liaison_client.product.get_option_details(request=option_details_req)
        return option_details_resp.option
    except RpcError as e:
        logger.error(f"Error fetching option details for {option_id}: {e}")
        return None


def user_msgs_to_price(user_messages):
    actual_price_tags = ["budget", "moderate", "mid-range", "luxury"]
    mapping = {v: v for v in actual_price_tags}
    mapping["cheap"] = "budget"
    mapping["affordable"] = "moderate"
    mapping["pricey"] = "luxury"
    mapping["fancy"] = "luxury"

    for msg in user_messages:
        for tag in actual_price_tags:
            if tag.lower() in msg.lower().split():
                logger.info(f"Applying brand price tag: {tag}")
                return tag
    return None


def user_msgs_to_age(user_messages):
    actual_age_tags = ["Boomer", "GenX", "Millenial", "GenZ"]
    mapping = {v: v for v in actual_age_tags}
    mapping["old"] = "Boomer"
    mapping["young"] = "GenZ"

    for msg in user_messages:
        for tag in actual_age_tags:
            if tag.lower() in msg.lower().split():
                logger.info(f"Applying brand age tag: {tag}")
                return tag
    return None


def user_msgs_to_aesthetic(user_messages):
    actual_aes = ['athleisure', 'avant garde', 'boho', 'casual', 'classic',
                  'country club', 'edgy', 'feminine', 'femme', 'formal',
                  'minimalist', 'modern', 'performance wear', 'romantic',
                  'sexy', 'sporty', 'streetwear', 'varied']
    mapping = {v: v for v in actual_aes}
    mapping['athletic'] = 'athleisure'
    mapping['athletic wear'] = 'athleisure'
    mapping['athleisure wear'] = 'athleisure'
    mapping['athleisurewear'] = 'athleisure'
    mapping['athleticwear'] = 'athleisure'
    mapping['athletic wear'] = 'athleisure'
    mapping['athletic wear'] = 'athleisure'
    mapping['avant-garde'] = 'avant garde'
    mapping['avantgarde'] = 'avant garde'
    mapping['bohemian'] = 'boho'
    mapping['bohemian chic'] = 'boho'
    mapping['bohemian style'] = 'boho'
    mapping['boho chic'] = 'boho'
    mapping['boho style'] = 'boho'

    for msg in user_messages:
        for tag in actual_aes:
            if tag.lower() in msg.lower().split():
                logger.info(f"Applying brand aesthetic tag: {tag}")
                return tag
    return None


def user_msgs_to_occassion(user_messages):
    actual_occasions = ['date night', 'event', 'everyday', 'exercising',
                        'lounging', 'resort', 'varies', 'work']
    mapping = {v: v for v in actual_occasions}
    mapping['varied'] = 'varies'
    mapping['workout'] = 'exercising'
    mapping['working out'] = 'exercising'
    mapping['working out'] = 'exercising'
    mapping['workout'] = 'exercising'

    for msg in user_messages:
        for tag in actual_occasions:
            if tag.lower() in msg.lower().split():
                logger.info(f"Applying brand occasion tag: {tag}")
                return tag



def es_prods_from_msgs(liaison_client, es_client,
                       inference_uri,
                       user_messages, user_id,
                       N=10):
    agent_resp = replay_chat(liaison_client, user_messages, user_id)
    chat_id = agent_resp.id
    listProductsReq = product_pb2.ListProductCardsRequest(
        page_size=10,
        chat_search_context=product_pb2.ListProductCardsSearchContext(
            chat_id=wrappers_pb2.StringValue(value=chat_id)  # pylint: disable=no-member
        )
    )
    listProductsReq.chat_search_context.search_settings.MergeFrom(agent_resp.search_settings)
    search_settings_dict = search_settings_proto_to_dict(agent_resp.search_settings)
    logger.debug(f"Search settings: {search_settings_dict}")

    image_embedding = None
    keywords = search_settings_dict['search_string']
    if 'image_search' in search_settings_dict:
        image_embedding = _inference(inference_uri, search_settings_dict['image_search'])
    else:
        image_embedding = _inference(inference_uri, search_settings_dict['keywords'])

    if 'lexical_search' in search_settings_dict:
        keywords = search_settings_dict['lexical_search']

    brand_ages = [user_msgs_to_age(user_messages)]
    if brand_ages[0] is None:
        brand_ages = None

    brand_prices = [user_msgs_to_price(user_messages)]
    if brand_prices[0] is None:
        brand_prices = None

    brand_aesthetics = [user_msgs_to_aesthetic(user_messages)]
    if brand_aesthetics[0] is None:
        brand_aesthetics = None

    brand_occasions = [user_msgs_to_occassion(user_messages)]
    if brand_occasions[0] is None:
        brand_occasions = None

    params = search_template_params(
        keywords=keywords,
        image_vector=image_embedding,
        brands=search_settings_dict.get("positive_filters", {}).get("brand", []),
        departments=search_settings_dict.get("positive_filters", {}).get("department", []),
        categories=search_settings_dict.get("positive_filters", {}).get("category", []),
        min_price=search_settings_dict["minimum_price"],
        max_price=search_settings_dict["maximum_price"],
        brand_ages=brand_ages,
        brand_prices=brand_prices,
        sizes=[],
    )

    matches = es_client.search_template(
        index='option',
        body={"id": "my-search-template", "params": params},
    )
    rendered = es_client.render_search_template(id="my-search-template",
                                                body={"params": params})['template_output']

    ranked_options = []
    for hit in matches['hits']['hits']:
        option_id = hit['_id']
        option = option_details(liaison_client, hit['_id'])
        main_image = None
        if option:
            for image in option.images:
                if image.is_main:
                    main_image = image.src
                    break
        ranked_options.append({
            "product_name": hit['_source']['name'],
            "brand_name": hit['_source']['brand_name'],
            "option_id": hit['_id'],
            'category': hit['_source']['category'],
            "main_image": main_image,
            "main_image_path": fetch_and_resize(main_image, option_id) if main_image else None,
            "product_description": hit['_source']['description'],
            "score": hit['_score']
        })
        if len(ranked_options) >= N:
            break

    df = pd.DataFrame(ranked_options)
    df.rename(columns={'option_id': 'id', 'product_name': 'name', 'product_description': 'description'}, inplace=True)
    return df, search_settings_dict, rendered


class ElasticsearchSearchBackend:
    """Use local experimental Elasticsearch index + template search for testing and prototyping."""
    def __init__(self, liaison_client, es_client, inference_uri, user_ids, name):
        self.liaison_client = liaison_client
        self.es_client = es_client
        self.inference_uri = inference_uri
        self.user_ids = user_ids
        self.name = name
        self.last_rendered = None

    def search(self, user_messages, dept):
        user_id = self.user_ids[dept]
        if isinstance(user_messages, str):
            user_messages = [user_messages]
        products, search_settings, rendered = es_prods_from_msgs(self.liaison_client, self.es_client,
                                                                 self.inference_uri,
                                                                 user_messages, user_id)
        products['backend'] = self.name
        self.last_rendered = rendered
        return products, search_settings


es_local_client = Elasticsearch("http://localhost:9200")
local_es = ElasticsearchSearchBackend(stag_liaison(), es_local_client,
                                      "http://localhost:10001/vectorize",
                                      stag_user_ids, 'local_es')
