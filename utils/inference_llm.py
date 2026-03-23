import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import Credentials

def inference_llm(model_id, params=None):
    model_id = model_id
    project_id = "a9913486-682d-4f0c-b006-5794d01c0913"
    default_params = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.2,
    }

    if params:
        default_params.update(params)

    return ModelInference(
        model_id=model_id,
        credentials=Credentials(
            url=os.environ.get("WATSONX_URL"),
            api_key=os.environ.get("WATSONX_APIKEY")
        ),
        project_id=project_id,
        params=default_params
    )