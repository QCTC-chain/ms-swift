# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import fields
from functools import partial
from typing import List, Union

from fastapi import File, UploadFile
import gradio as gr
from packaging import version
from transformers.utils import strtobool

import swift
from swift.llm import DeployArguments, EvalArguments, ExportArguments, RLHFArguments, SwiftPipeline, WebUIArguments
from swift.ui.llm_eval.llm_eval import LLMEval
from swift.ui.llm_export.llm_export import LLMExport
from swift.ui.llm_infer.llm_infer import LLMInfer
from swift.ui.llm_train.llm_train import LLMTrain

locale_dict = {
    'title': {
        'zh': '🚀SWIFT: 轻量级大模型训练推理框架',
        'en': '🚀SWIFT: Scalable lightWeight Infrastructure for Fine-Tuning and Inference'
    },
    'sub_title': {
        'zh':
        '请查看 <a href=\"https://github.com/modelscope/swift/tree/main/docs/source\" target=\"_blank\">'
        'SWIFT 文档</a>来查看更多功能，使用SWIFT_UI_LANG=en环境变量来切换英文界面',
        'en':
        'Please check <a href=\"https://github.com/modelscope/swift/tree/main/docs/source_en\" target=\"_blank\">'
        'SWIFT Documentation</a> for more usages, Use SWIFT_UI_LANG=zh variable to switch to Chinese UI',
    },
    'star_beggar': {
        'zh':
        '喜欢<a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>就动动手指给我们加个star吧🥺 ',
        'en':
        'If you like <a href=\"https://github.com/modelscope/swift\" target=\"_blank\">SWIFT</a>, '
        'please take a few seconds to star us🥺 '
    },
}


class SwiftWebUI(SwiftPipeline):

    args_class = WebUIArguments
    args: args_class

    def run(self):
        lang = os.environ.get('SWIFT_UI_LANG') or self.args.lang
        share_env = os.environ.get('WEBUI_SHARE')
        share = strtobool(share_env) if share_env else self.args.share
        server = os.environ.get('WEBUI_SERVER') or self.args.host
        port_env = os.environ.get('WEBUI_PORT')
        port = int(port_env) if port_env else self.args.port
        is_gradio_app = self.args.model or self.args.ckpt_dir
        LLMTrain.set_lang(lang)
        LLMInfer.set_lang(lang)
        LLMExport.set_lang(lang)
        LLMEval.set_lang(lang)
        with gr.Blocks(title='SWIFT WebUI') as app:
            if is_gradio_app:
                gr.HTML(f'<h1><center>{self.args.studio_title}</center></h1>')
            else:
                try:
                    _version = swift.__version__
                except AttributeError:
                    _version = ''
                gr.HTML(f"<h1><center>{locale_dict['title'][lang]}({_version})</center></h1>")
                gr.HTML(f"<h3><center>{locale_dict['sub_title'][lang]}</center></h3>")
            with gr.Tabs():
                if is_gradio_app:
                    if self.args.ckpt_dir:
                        self.args.model = self.args.ckpt_dir
                    for f in fields(self.args):
                        if getattr(self.args, f.name):
                            LLMInfer.default_dict[f.name] = getattr(self.args, f.name)
                    LLMInfer.is_gradio_app = True
                    LLMInfer.is_multimodal = self.args.model_meta.is_multimodal
                    LLMInfer.build_ui(LLMInfer)
                else:
                    LLMTrain.build_ui(LLMTrain)
                    LLMInfer.build_ui(LLMInfer)
                    LLMExport.build_ui(LLMExport)
                    LLMEval.build_ui(LLMEval)

            concurrent = {}
            if version.parse(gr.__version__) < version.parse('4.0.0'):
                concurrent = {'concurrency_count': 5}
            if is_gradio_app:
                from swift.utils import find_free_port
                LLMInfer.element('port').value = str(find_free_port())
                app.load(LLMInfer.deploy_model, list(LLMInfer.valid_elements().values()),
                         [LLMInfer.element('runtime_tab'),
                          LLMInfer.element('running_tasks')])
            else:
                app.load(
                    partial(LLMTrain.update_input_model, arg_cls=RLHFArguments),
                    inputs=[LLMTrain.element('model')],
                    outputs=[LLMTrain.element('train_record')] + list(LLMTrain.valid_elements().values()))
                app.load(
                    partial(LLMInfer.update_input_model, arg_cls=DeployArguments, has_record=False),
                    inputs=[LLMInfer.element('model')],
                    outputs=list(LLMInfer.valid_elements().values()))
                app.load(
                    partial(LLMExport.update_input_model, arg_cls=ExportArguments, has_record=False),
                    inputs=[LLMExport.element('model')],
                    outputs=list(LLMExport.valid_elements().values()))
                app.load(
                    partial(LLMEval.update_input_model, arg_cls=EvalArguments, has_record=False),
                    inputs=[LLMEval.element('model')],
                    outputs=list(LLMEval.valid_elements().values()))
        
        fastApi, local_url, share_url = app.queue(**concurrent).launch(
            server_name=server, 
            inbrowser=True, 
            server_port=port, 
            height=800, 
            share=share, 
            prevent_thread_lock=True)

        self.registry_external_api_request(fastApi)

        app.block_thread()
        return fastApi, local_url, share_url
    
    def registry_external_api_request(self, fastApi):
        @fastApi.get("/external_api/invoke/show_log")
        async def show_log(
            logging_dir: str = None,
            file_name: str = None,
            log_file: str = None,
            offset: int = 0):
            import collections
            import os.path

            def trained_percent(line):
                parts = line.split('Train:')
                if len(parts) > 1:
                    percent_str = parts[1].split('|')[0].strip()
                    if percent_str.endswith('%'):
                        percent_value = percent_str[:-1]
                        return percent_value
                return None

            if not log_file:
                log_file = os.path.join(logging_dir, file_name)

            if not os.path.isfile(log_file):
                raise Exception(f'必须指定 log_file 或同时指定 logging_dir 和 file_name')

            maxlen = int(os.environ.get('MAX_LOG_LINES', 50))
            lines = collections.deque(maxlen=maxlen)
            trained_process = None
            try:
                with open(log_file, 'r', encoding='utf-8') as input:
                    # Skip lines until start_line
                    for _ in range(offset):
                        next(input, None)

                    # Read the next num_lines lines
                    for _ in range(maxlen):
                        line = input.readline()
                        if not line:
                            break
                        if line.startswith('Train:'):
                            trained_process = trained_percent(line)
                        lines.append(line)
                    return {'data': lines,'trained': trained_process, 'next': len(lines)}
            except IOError:
                pass
        
        @fastApi.get("/external_api/invoke/list_models")
        async def support_model_list():
            from swift.llm.model.register import get_all_models_with_filter
            from swift.llm.model.constant import MLLMModelType
            model_type_filter = ['qwen','qwen2_moe', 'qwq', 'codefuse_qwen', 'modelscope_agent', 'marco_o1',
                      'llama', 'reflection', 'megrez', 'sus', 'codefuse_codellama', 'mengzi3', 'ziya',
                      'numina', 'atom', 'chatglm2', 'glm_edge', 'codefuse_codegeex2', 'codegeex4',
                      'longwriter_llama3_1', 'internlm', 'internlm2', 'deepseek', 'deepseek_moe', 'deepseek_v2',
                      'openbuddy_llama', 'openbuddy_llama3', 'openbuddy_mistral', 'openbuddy_mixtral',
                      'baichuan', 'minicpm', 'minicpm_chatml', 'minicpm3', 'minicpm_moe', 'telechat',
                      'telechat2', 'telechat2_115b', 'mistral', 'zephyr', 'mixtral', 'mistral_nemo', 'wizardlm2',
                      'wizardlm2_moe', 'phi2', 'phi2_small', 'phi3', 'phi3_moe', 'gemma', 'skywork', 'skywork_o1',
                      'orion', 'xverse', 'xverse_moe', 'seggpt', 'bluelm', 'c4ai', 'dbrx', 'grok', 'mamba', 'polylm',
                      'aya', 'yuan2', 'phi3_small', 'gemma2', 'chatglm3', 'yi_coder', 'llama3', 'llama3_1', 'qwen2']
            # 过滤所有的多模态大模型
            for k in MLLMModelType.__dict__:
                if k.startswith('__'):
                    continue
                value = getattr(MLLMModelType, k)
                if isinstance(value, str):
                    model_type_filter.append(value)

            model_id_filter = [
                            'Qwen/Qwen1.5-0.5B-Chat',
                            'Qwen/Qwen1.5-1.8B-Chat',
                            'Qwen/Qwen1.5-4B-Chat',
                            'Qwen/Qwen1.5-7B-Chat',
                            'Qwen/Qwen1.5-14B-Chat',
                            'Qwen/Qwen1.5-32B-Chat',
                            'Qwen/Qwen1.5-72B-Chat',
                            'Qwen/Qwen1.5-110B-Chat',
                            'Qwen/Qwen1.5-0.5B',
                            'Qwen/Qwen1.5-1.8B',
                            'Qwen/Qwen1.5-4B',
                            'Qwen/Qwen1.5-7B',
                            'Qwen/Qwen1.5-14B',
                            'Qwen/Qwen1.5-32B',
                            'Qwen/Qwen1.5-72B',
                            'Qwen/Qwen1.5-110B',
                            'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-4B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-32B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-72B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-110B-Chat-GPTQ-Int4',
                            'Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8',
                            'Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8',
                            'Qwen/Qwen1.5-4B-Chat-GPTQ-Int8',
                            'Qwen/Qwen1.5-7B-Chat-GPTQ-Int8',
                            'Qwen/Qwen1.5-14B-Chat-GPTQ-Int8',
                            'Qwen/Qwen1.5-72B-Chat-GPTQ-Int8',
                            'Qwen/Qwen1.5-0.5B-Chat-AWQ',
                            'Qwen/Qwen1.5-1.8B-Chat-AWQ',
                            'Qwen/Qwen1.5-4B-Chat-AWQ',
                            'Qwen/Qwen1.5-7B-Chat-AWQ',
                            'Qwen/Qwen1.5-14B-Chat-AWQ',
                            'Qwen/Qwen1.5-32B-Chat-AWQ',
                            'Qwen/Qwen1.5-72B-Chat-AWQ',
                            'Qwen/Qwen1.5-110B-Chat-AWQ',
                            'Qwen/CodeQwen1.5-7B',
                            'Qwen/CodeQwen1.5-7B-Chat',
                            'Qwen/CodeQwen1.5-7B-Chat-AWQ',
                            'Qwen/Qwen2-0.5B-Instruct',
                            'Qwen/Qwen2-1.5B-Instruct',
                            'Qwen/Qwen2-7B-Instruct',
                            'Qwen/Qwen2-72B-Instruct',
                            'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2-72B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2-7B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2-72B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2-0.5B-Instruct-AWQ',
                            'Qwen/Qwen2-1.5B-Instruct-AWQ',
                            'Qwen/Qwen2-7B-Instruct-AWQ',
                            'Qwen/Qwen2-72B-Instruct-AWQ',
                            'Qwen/Qwen2-Math-1.5B-Instruct',
                            'Qwen/Qwen2-Math-7B-Instruct',
                            'Qwen/Qwen2-Math-72B-Instruct',
                            'Qwen/Qwen2.5-0.5B-Instruct',
                            'Qwen/Qwen2.5-1.5B-Instruct',
                            'Qwen/Qwen2.5-3B-Instruct',
                            'Qwen/Qwen2.5-14B-Instruct',
                            'Qwen/Qwen2.5-32B-Instruct',
                            'Qwen/Qwen2.5-72B-Instruct',
                            'Qwen/Qwen2.5-0.5B',
                            'Qwen/Qwen2.5-1.5B',
                            'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-0.5B-Instruct-AWQ',
                            'Qwen/Qwen2.5-1.5B-Instruct-AWQ',
                            'Qwen/Qwen2.5-3B-Instruct-AWQ',
                            'Qwen/Qwen2.5-7B-Instruct-AWQ',
                            'Qwen/Qwen2.5-14B-Instruct-AWQ',
                            'Qwen/Qwen2.5-32B-Instruct-AWQ',
                            'Qwen/Qwen2.5-72B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Math-1.5B-Instruct',
                            'Qwen/Qwen2.5-Math-7B-Instruct',
                            'Qwen/Qwen2.5-Math-72B-Instruct',
                            'Qwen/Qwen2.5-Math-1.5B',
                            'Qwen/Qwen2.5-Math-7B',
                            'Qwen/Qwen2.5-Math-72B',
                            'Qwen/Qwen2.5-Coder-0.5B-Instruct',
                            'Qwen/Qwen2.5-Coder-1.5B-Instruct',
                            'Qwen/Qwen2.5-Coder-3B-Instruct',
                            'Qwen/Qwen2.5-Coder-7B-Instruct',
                            'Qwen/Qwen2.5-Coder-14B-Instruct',
                            'Qwen/Qwen2.5-Coder-32B-Instruct',
                            'Qwen/Qwen2.5-Coder-0.5B',
                            'Qwen/Qwen2.5-Coder-1.5B',
                            'Qwen/Qwen2.5-Coder-3B',
                            'Qwen/Qwen2.5-Coder-7B',
                            'Qwen/Qwen2.5-Coder-14B',
                            'Qwen/Qwen2.5-Coder-32B',
                            'Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Coder-3B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Coder-7B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Coder-14B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Coder-32B-Instruct-AWQ',
                            'Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8',
                            'Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4',
                            'Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8',
                            "LLM-Research/Meta-Llama-3-8B-Instruct",
                            "LLM-Research/Meta-Llama-3-70B-Instruct",
                            "swift/Meta-Llama-3-8B-Instruct-GPTQ-Int4",
                            "swift/Meta-Llama-3-8B-Instruct-GPTQ-Int8",
                            "swift/Meta-Llama-3-8B-Instruct-AWQ",
                            "swift/Meta-Llama-3-70B-Instruct-GPTQ-Int4",
                            "swift/Meta-Llama-3-70B-Instruct-GPTQ-Int8",
                            "swift/Meta-Llama-3-70B-Instruct-AWQ",
                            "LLM-Research/Meta-Llama-3.1-8B-Instruct",
                            "LLM-Research/Meta-Llama-3.1-70B-Instruct",
                            "LLM-Research/Meta-Llama-3.1-405B-Instruct",
                            "LLM-Research/Meta-Llama-3.1-70B-Instruct-FP8",
                            "LLM-Research/Meta-Llama-3.1-405B-Instruct-FP8",
                            "LLM-Research/Meta-Llama-3.1-8B-Instruct-BNB-NF4",
                            "LLM-Research/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
                            "LLM-Research/Meta-Llama-3.1-405B-Instruct-BNB-NF4",
                            "LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
                            "LLM-Research/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
                            "LLM-Research/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4",
                            "LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                            "LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                            "LLM-Research/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
                            "AI-ModelScope/Llama-3.1-Nemotron-70B-Instruct-HF",
                            "LLM-Research/Llama-3.2-1B",
                            "LLM-Research/Llama-3.2-1B-Instruct",
                            "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
                            "01ai/Yi-6B",
                            "01ai/Yi-6B-200K",
                            "01ai/Yi-6B-Chat",
                            "01ai/Yi-6B-Chat-4bits",
                            "01ai/Yi-6B-Chat-8bits",
                            "01ai/Yi-34B-Chat",
                            "01ai/Yi-34B-Chat-4bits",
                            "01ai/Yi-34B-Chat-8bits",
                            "01ai/Yi-1.5-6B-Chat",
                            "01ai/Yi-1.5-9B-Chat",
                            "01ai/Yi-1.5-9B-Chat-16K",
                            "01ai/Yi-1.5-34B-Chat",
                            "01ai/Yi-1.5-34B-Chat-16K",
                            "AI-ModelScope/Yi-1.5-6B-Chat-GPTQ",
                            "AI-ModelScope/Yi-1.5-6B-Chat-AWQ",
                            "AI-ModelScope/Yi-1.5-9B-Chat-GPTQ",
                            "AI-ModelScope/Yi-1.5-9B-Chat-AWQ",
                            "AI-ModelScope/Yi-1.5-34B-Chat-GPTQ",
                            "AI-ModelScope/Yi-1.5-34B-Chat-AWQ",
                            "01ai/Yi-Coder-1.5B-Chat",
                            "01ai/Yi-Coder-9B-Chat"]

            return get_all_models_with_filter(model_type_filter=model_type_filter, model_id_filter=model_id_filter)

        @fastApi.get("/external_api/invoke/list_gpus")
        async def list_gpus():
            import torch

            gpu_count = 0
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
            return [str(i) for i in range(gpu_count)] + ['cpu']

        @fastApi.get("/external_api/invoke/list_quant_methods")
        async def support_quant_list():
            quant_methods = ['gptq', 'aqlm', 'awq']
            return quant_methods

        @fastApi.get("/external_api/invoke/list_quant_bits")
        async def support_quant_bits_list():
            quant_bits = [1, 2, 3, 4, 8]
            return quant_bits

        @fastApi.get("/external_api/invoke/model_meta")
        async def get_model_meta(model_id: str):
            from swift.llm import TEMPLATE_MAPPING,BaseArguments
            from swift.llm.model.register import get_matched_model_meta

            model_meta = get_matched_model_meta(model_id)
            if model_meta:
                return {
                    'model_type': model_meta.model_type, 
                    'model_template': model_meta.template,
                    'model_system': TEMPLATE_MAPPING[model_meta.template].default_system
                }
            
            local_args_path = os.path.join(model_id, 'args.json')
            if os.path.exists(local_args_path):
                args = BaseArguments(ckpt_dir=model_id, load_data_args=True)
                template = getattr(args, 'template', None)
                return {
                    'model_type': getattr(args, 'model_type', None), 
                    'model_template': template,
                    'model_system': TEMPLATE_MAPPING[template].default_system
                }
            

        @fastApi.post("/external_api/invoke/uploadfile")
        async def upload_file(dataset_file: UploadFile = File(...)):
            try:
                import os
                import uuid
                import tempfile
                import shutil
                from datetime import datetime

                dataset_path = os.path.join("sft_dataset", 'user_assistant')
                if not os.path.exists(dataset_path):
                        os.makedirs(dataset_path)
                
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=os.path.join('sft_dataset', 'user_assistant'))
                with os.fdopen(tmp_fd, "wb") as tmp:
                    tmp.write(await dataset_file.read())
                
                dst = os.path.join('sft_dataset', 'user_assistant', f'{uuid.uuid4()}_{dataset_file.filename}')
                shutil.move(tmp_path, dst)
                return {'location': os.path.abspath(dst)}
            except Exception as e:
                raise e

def webui_main(args: Union[List[str], WebUIArguments, None] = None):
    return SwiftWebUI(args).main()
