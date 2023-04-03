from flask import Flask
from flask import make_response
from flask import request
from flask import send_from_directory
import threading
import portpicker
from IPython import display
from IPython.utils import io
import sys
import os
import shutil
from datetime import datetime
import json
import logging
import requests
import zipfile
from google.colab import output
from google.colab import _message

# Disable logging from werkzeug.
#
# Without this, flask will show a warning about using dev server (which is OK
# in our usecase).
logging.getLogger('werkzeug').disabled = True

def js(script):
  display.display(display.Javascript(script))

def html(script):
  display.display(display.HTML(script))

def Server(inference_fn, height = 900, tmp_dir = '/tmp'):
  """Creates a server that serves visual blocks web app in an iFrame.
  
  Other than serving the web app, it will also listen to requests sent from the
  web app at the '/apipost/inference' endpoint. Once a request is received, it
  will use the tensor data in the request body to call the 'inference_fn' which
  in turn will run the model in Colab and return the result tensors to the
  web app. 

  Args:
    inference_fn: An python function defined in the same colab notebook that
      does the actual model inference. The function should accept a 'tensors'
      array, and each element of the array is a dict:
      {'tensorValues': <values_array>, 'tensorShape': <shape_array>}. The
      function should return a dict {'tensors': <tensors_array>} where the
      <tensors_array> has the same format as the function parameter. This
      function will be called when the web app sends a request to the
      '/apipost/inference' endpoint.
    height: The height of the embedded iFrame.
    tmp_dir: The tmp dir where the server stores the web app's static resources.
  """

  app = Flask(__name__)

  # Disable startup messages.
  cli = sys.modules['flask.cli']
  cli.show_server_banner = lambda *x: None

  # Prepare tmp dir and log file.
  base_path = tmp_dir + '/visual-blocks-colab';
  if os.path.exists(base_path):
    shutil.rmtree(base_path)
  os.mkdir(base_path)
  log_file_path = base_path + '/log'
  open(log_file_path, 'w').close()

  # Download the zip file that bundles the visual blocks web app.
  bundle_target_path = os.path.join(base_path, 'visual_blocks.zip')
  url = 'https://storage.googleapis.com/tfweb/rapsai-colab-bundles/visual_blocks_1680551188.zip'
  r = requests.get(url)
  with open(bundle_target_path , 'wb') as zip_file:
    zip_file.write(r.content)

  # Unzip it.
  # This will unzip all files to {base_path}/build.
  with zipfile.ZipFile(bundle_target_path, 'r') as zip_ref:
    zip_ref.extractall(base_path)
  site_root_path = os.path.join(base_path, 'build')

  def log(msg):
    """Logs the given message to the log file."""
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    with open(log_file_path, "a") as log_file:
      log_file.write("{}: {}\n".format(dt_string, msg))

  # Note: using "/api/..." for POST requests is not allowed. 
  @app.route('/apipost/inference', methods=['POST'])
  def inference():
    """Handler for the api endpoint."""
    tensors = request.json['tensors']
    result = {}
    try:
      result = inference_fn(tensors)
    except Exception as e:
      result = {'error': str(e)}
    finally:
      resp = make_response(json.dumps(result))
      resp.headers['Content-Type'] = 'application/json'
      return resp

  @app.route('/<path:path>')
  def get_static(path):
    """Handler for serving static resources."""
    return send_from_directory(site_root_path, path)
      
  def embed(port, height):
    """Embeds iFrame."""
    shell = """
      (async () => {
            // Listen to event when user clicks the "Save to colab" button.
            window.addEventListener('message', (e) => {
              if (!e.data) {
                return;
              }
              const cmd = e.data.cmd;
              if (cmd === 'rapsai-save-to-colab') {
                const project = e.data.data;
                google.colab.kernel.invokeFunction('saveProject', [JSON.stringify(project)], {});
              }
            });

            const url = await google.colab.kernel.proxyPort(%PORT%, {"cache": true}) 
                + 'index.html#/edit/_%PIPELINE_JSON%';
            const iframe = document.createElement('iframe');
            iframe.src = url;
            iframe.setAttribute('width', '100%');
            iframe.setAttribute('height', '%HEIGHT%');
            iframe.setAttribute('frameborder', 0);
            iframe.setAttribute('style', 'border: 1px solid #ccc; box-sizing: border-box;');
            iframe.setAttribute('allow', 'camera;microphone');
            
            const uiContainer = document.body.querySelector('#ui-container');
            uiContainer.innerHTML = '';
            if (google.colab.kernel.accessAllowed) {
              uiContainer.appendChild(iframe);
            }
        })();
    """
    replacements = [
        ("%PORT%", "%d" % port),
        ("%HEIGHT%", "%d" % height),
    ]
    # Append pipeline_json string to the url if it is saved in the notebook.
    if pipeline_json == '':
      replacements.append(('%PIPELINE_JSON%', ''))
    else:
      replacements.append(('%PIPELINE_JSON%', '?project=%s' % pipeline_json))
    
    for (k, v) in replacements:
        shell = shell.replace(k, v)

    js(shell)
    
  def show_app():
    """Shows the web app."""
    embed(port, height)

  def save_project(data):
    """Puts the given project json data into the given div.

    The content of the div will be persisted in notebook.    
    """
    with output.redirect_to_element('#pipeline-output'):
      js('document.body.querySelector("#pipeline-output").innerHTML = ""')
      html(data)

  def show_controls():
    html('''<style>
    #pipeline-output-title {
      margin-top: 12px;
    }

    #pipeline-output {
      color: #999;
      font-size: 11px;
      margin: 4px 0;
      max-height: 36px;
      overflow-y: auto;
      margin-bottom: 12px;
      background-color: #f9f9f9;
      border: 1px solid #ccc;
      padding: 8px;
      border-radius: 4px;
    }

    #pipeline-message {
      font-size: 14px;
      padding: 8px;
      background-color: #ffefe1;
      color: #99730a;
      border: 1px solid #99730a;
      border-radius: 4px;
      width: fit-content;
    }
  </style>''')
    html('<div id="pipeline-output-title">Saved pipeline:</div>')
    html('<div id="pipeline-output">N/A</div>')
    html('<div id="pipeline-message"></div>')
    js('''
    const msgEle = document.querySelector('#pipeline-message');
    if (!google.colab.kernel.accessAllowed) {
      msgEle.style.display = 'block';
      msgEle.textContent = 'ⓘ To start, run the cell above with `Server` first then run this cell.'
    } else {
      msgEle.style.display = 'none';
      google.colab.kernel.invokeFunction('showApp', [], {});
    }
  ''')
    html('<div id="ui-container"></div>')

  # Read the current notebook and find the pipeline json.
  pipeline_json = ''
  notebook_json_string = _message.blocking_request('get_ipynb', request='', timeout_sec=60)
  for cell in notebook_json_string['ipynb']['cells']:
    if 'outputs' not in cell:
      continue
    for cur_output in cell['outputs']:
      if 'data' not in cur_output:
        continue
      if 'text/html' not in cur_output['data']:
        continue
      if cur_output['data']['text/html'] is not None:
        cur_text = cur_output['data']['text/html']
        if cur_text[0].startswith('{"project":'):
          pipeline_json = cur_text[0]

  # Register callback for saving project.
  output.register_callback('saveProject', save_project)
  output.register_callback('showApp', show_app)

  # Start background server.
  port = portpicker.pick_unused_port()
  threading.Thread(target=app.run, kwargs={'host':'::','port':port}).start()

  # A thin wrapper class for exposing a "show_ui" method.
  class _Server:
    def show_ui(self):
      show_controls()
      if pipeline_json:
        save_project(pipeline_json)
      
  return _Server()
