import os
import shutil
import tarfile
import numpy as np
import xmltodict


def load_otb(file_path, tmp_dir="tmpopen"):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    if file_path.endswith(".zip"):
        shutil.unpack_archive(file_path, tmp_dir)
    else:
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=tmp_dir)

    signals = [f for f in os.listdir(tmp_dir) if f.endswith('.sig')]
    if not signals:
        raise FileNotFoundError("No .sig file found")

    sig_file = signals[0]
    xml_file = sig_file.replace('.sig', '.xml')
    xml_path = os.path.join(tmp_dir, xml_file)

    with open(xml_path) as f:
        meta = xmltodict.parse(f.read())

    device_name = meta['Device']['@Name']
    fs = float(meta['Device']['@SampleFrequency'])
    ad_bits = int(meta['Device']['@ad_bits'])

    adapters = meta['Device']['Channels']['Adapter']
    if not isinstance(adapters, list):
        adapters = [adapters]

    n_channels = sum(
        len(a['Channel']) if isinstance(a['Channel'], list) else 1
        for a in adapters
    )

    with open(os.path.join(tmp_dir, sig_file), 'rb') as f:
        dtype = np.int16 if ad_bits == 16 else np.int32
        raw = np.fromfile(f, dtype=dtype)
        data = raw.reshape((n_channels, -1), order='F').astype(np.float64)

    time = np.arange(data.shape[1]) / fs

    shutil.rmtree(tmp_dir)

    return data, time, fs, device_name