import gradio as gr
import os
import argparse
from library.train_avatar_gui import train_avatar_tab
from pathlib import Path
import subprocess


# def setup():
#     install_cmds = [
#         ['pip', 'install', 'gradio'],
#         ['pip', 'install', 'open_clip_torch'],
#         ['pip', 'install', 'clip-interrogator==0.6.0'],
#     ]
#     for cmd in install_cmds:
#         print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

# setup()

def UI(**kwargs):
    css = ''

    if os.path.exists('./style.css'):
        with open(os.path.join('./style.css'), 'r', encoding='utf8') as file:
            print('Load CSS...')
            css += file.read() + '\n'

    interface = gr.Blocks(css=css, title='MoonShot Trainer GUI', theme=gr.themes.Default())

    root_path = Path().resolve()
    train_project_folder = r"%s\train_project" % (root_path)
    os.makedirs(train_project_folder, exist_ok=True)

    with interface:
        with gr.Tab('Avatar Trainer'):
            train_avatar_tab()


    # Show the interface
    launch_kwargs = {}
    username = kwargs.get('username')
    password = kwargs.get('password')
    server_port = kwargs.get('server_port', 0)
    inbrowser = kwargs.get('inbrowser', False)
    share = kwargs.get('share', False)
    server_name = kwargs.get('listen')

    launch_kwargs['server_name'] = server_name
    if username and password:
        launch_kwargs['auth'] = (username, password)
    if server_port > 0:
        launch_kwargs['server_port'] = server_port
    if inbrowser:
        launch_kwargs['inbrowser'] = inbrowser
    if share:
        launch_kwargs['share'] = share
    interface.launch(**launch_kwargs)


if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.48)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=3333,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    UI(
        username=args.username,
        password=args.password,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        share=args.share,
        listen=args.listen,
    )
