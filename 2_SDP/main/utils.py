from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont
from transformers import PretrainedConfig
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 
import imageio.v2 as imageio
from torch import nn 
import numpy as np 
import textwrap 
import pickle 
import torch 
import copy 
import os 
import math


def prepare_images_for_saving(images_tensor, resolution, grid_size=4, range_type="neg1pos1"):
    # Accept [B,3,H,W] or [3,H,W]
    if images_tensor.dim() == 3:
        images_tensor = images_tensor.unsqueeze(0)  # -> [1,3,H,W]

    # Map to [0,255] only if not already uint8
    if range_type != "uint8":
        images = (images_tensor * 0.5 + 0.5).clamp(0, 1) * 255.0
    else:
        images = images_tensor

    # Crop spatially to the requested resolution (defensive)
    images = images[:, :, :resolution, :resolution]  # [B,3,res,res]

    # Dynamic grid: minimal square big enough for B
    B = images.shape[0]
    g = int(math.ceil(math.sqrt(max(B, 1))))  # at least 1x1
    nslots = g * g

    # Pad to exactly g^2 tiles (no truncation unless you want it)
    if B < nslots:
        pad = images.new_zeros(nslots - B, images.shape[1], resolution, resolution)
        images = torch.cat([images, pad], dim=0)
    elif B > nslots:
        images = images[:nslots]  # optional: truncate extras; remove if you prefer a bigger grid

    # Channel-last for numpy grid
    images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")  # [nslots,res,res,3]

    # Fold into a g x g grid
    grid = images.reshape(g, g, resolution, resolution, 3)
    grid = np.swapaxes(grid, 1, 2).reshape(g * resolution, g * resolution, 3)
    return grid


def prepare_debug_output(tensor, resolution):
    # N x T x 3 x H x W 
    N, T = tensor.shape[:2]
    tensor = tensor.transpose(0, 1)
    tensor = ((tensor * 0.5 + 0.5).clamp(0, 1) * 255).permute(0, 1, 3, 4, 2).detach().cpu().numpy().astype("uint8")      
    tensor = np.swapaxes(tensor, 1, 2).reshape(T*resolution, N*resolution, 3)
    return tensor 

def draw_valued_array(data, output_dir, grid_size=4):
    fig = plt.figure(figsize=(20,20))

    # Flatten to 1D numpy
    flat = data.detach().cpu().numpy().ravel() if torch.is_tensor(data) else np.asarray(data).ravel()

    # Smallest square grid that fits N values
    g = int(math.ceil(math.sqrt(max(flat.size, 1))))

    # Pad with zeros to exactly g*g, then reshape
    if flat.size < g * g:
        flat = np.pad(flat, (0, g * g - flat.size), mode="constant")
    data = flat.reshape(g, g)

    # Make downstream use the dynamic grid size
    grid_size = g
    cax = plt.matshow(data, cmap='viridis')  # Change cmap to your desired color map
    plt.colorbar(cax)

    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='black')

    plt.savefig(os.path.join(output_dir, "cache.jpg"))
    plt.close('all')

    # read the image 
    image = imageio.imread(os.path.join(output_dir, "cache.jpg"))
    return image

def draw_probability_histogram(data):
    fig = plt.figure(figsize=(5,5))

    plt.hist(data, color='blue', edgecolor='black')
    plt.title('Histogram of Realism Prediction')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the canvas as a PIL image
    image = Image.frombytes(
        "RGB", canvas.get_width_height(), canvas.tostring_rgb()
    )
    plt.close('all')
    return image

def draw_gradient_norm(data, pred_realism, num_bin=10, bin_size=0.1):
    mean_list = [] 
    for bin_idx in range(num_bin):
        start = bin_idx * bin_size
        end = (bin_idx + 1) * bin_size
        data_bin = data[(pred_realism >= start) & (pred_realism < end)]

        if len(data_bin) == 0:
            mean_list.append(0)
        else:
            mean_list.append(data_bin.mean())
        
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(num_bin) * bin_size, mean_list)
    plt.title('Gradient Norm')
    plt.xlabel('Predicted Realism')
    plt.ylabel('Mean Grad Norm')

    plt.xlim(0, 1)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the canvas as a PIL image
    image = Image.frombytes(
        "RGB", canvas.get_width_height(), canvas.tostring_rgb()
    )
    plt.close('all')
    return image

def draw_array(indices, values, min_val=None, max_val=None):
    fig = plt.figure(figsize=(5,5))
    plt.plot(indices, values)

    if max_val is None: 
        max_val = max(values[values!= 1.0].max() * 1.1, 0.05)
    
    if min_val is None: 
        min_val = 0 

    plt.ylim(min_val, max_val)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the canvas as a PIL image
    image = Image.frombytes(
        "RGB", canvas.get_width_height(), canvas.tostring_rgb()
    )
    plt.close('all')
    return image

def cycle(dl):
    while True:
        for data in dl:
            yield data

def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

class EMA(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        self.decay = decay

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # update the parameters
        update_ema(self.ema_model.parameters(), model.parameters(), self.decay)

        # update the buffers with certain exception 
        for (buffer_ema_name, buffer_ema), (buffer_name, buffer) in zip(self.ema_model.named_buffers(), model.named_buffers()):
            if "num_batches_tracked" in buffer_ema_name:
                buffer_ema.copy_(buffer)
            else:
                update_ema([buffer_ema], [buffer], self.decay)

def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, shape, row_index):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    #with lmdb_env.begin() as txn:
    #    row_bytes = txn.get(data_key)

    #array = np.frombuffer(row_bytes, dtype=dtype)
    
    #if len(shape) > 0:
    #    array = array.reshape(shape)
    #return array 
    """
    Robustly load one row from LMDB, accepting both zero-padded and non-padded
    keys (e.g., images_000123_data or images_123_data).
    """
    with lmdb_env.begin(write=False) as txn:
        tried, row_bytes = [], None
        for pad in (6, 5, 4, 3, 2, 1, 0):
            key_str = (f"{array_name}_{row_index:0{pad}d}_data" if pad
                       else f"{array_name}_{row_index}_data")
            row_bytes = txn.get(key_str.encode("utf-8"))
            tried.append(key_str)
            if row_bytes is not None:
                break
        if row_bytes is None:
            raise KeyError(
                f"LMDB row not found for {array_name}[{row_index}]. Tried:\n" +
                "\n".join(tried)
            )
        arr = np.frombuffer(row_bytes, dtype=dtype)
        return arr.reshape(shape)
    
 ###debug decode image shape ####
 #def get_array_shape_from_lmdb(lmdb_env, array_name):
 #   with lmdb_env.begin() as txn:
 #       image_shape = txn.get(f"{array_name}_shape".encode()).decode()
 #       image_shape = tuple(map(int, image_shape.split()))

  #  retur# utils.py
import os, json, re
import lmdb
from io import BytesIO
from PIL import Image
import numpy as np

def _decode_shape(val_bytes):
    s = val_bytes.decode()
    # try JSON first (e.g., "[256,256,3]")
    if s.strip().startswith("["):
        return tuple(json.loads(s))
    # fallback "H,W,C" or "C,H,W"
    parts = re.split(r"[,\s]+", s.strip())
    return tuple(int(x) for x in parts if x)

def _infer_shape_from_first_sample(txn, array_name):
    """Find first key like f"{array_name}_000000", read, infer shape."""
    prefix = f"{array_name}_".encode()
    cur = txn.cursor()
    for k, v in cur:
        if not k.startswith(prefix):
            continue
        # Try numpy raw array first
        try:
            arr = np.load(BytesIO(v), allow_pickle=False)
            if hasattr(arr, "shape"):
                return tuple(map(int, arr.shape))
        except Exception:
            pass
        # Try image bytes
        try:
            img = Image.open(BytesIO(v)).convert("RGB")
            return (img.height, img.width, 3)
        except Exception:
            pass
        break
    return None

def get_array_shape_from_lmdb(env: lmdb.Environment, array_name: str, write_back=True):
    with env.begin(write=False) as txn:
        shape_key = f"{array_name}_shape".encode()
        val = txn.get(shape_key)
        if val is not None:
            return _decode_shape(val)

    # Not found: diagnose and optionally self-heal
    with env.begin(write=False) as txn:
        # list available *_shape keys to help debugging
        cur = txn.cursor()
        shapes_available = []
        count = 0
        for k, _v in cur:
            if k.endswith(b"_shape"):
                try:
                    shapes_available.append(k.decode())
                    count += 1
                except Exception:
                    pass
            if count >= 50:
                break

    # Try to infer from a sample and write back
    inferred = _infer_shape_from_first_sample(txn, array_name)
    if inferred is not None and write_back:
        # write the inferred shape so future calls are fast
        with env.begin(write=True) as wtxn:
            wtxn.put(f"{array_name}_shape".encode(), json.dumps(list(inferred)).encode())
        return inferred

    # If we reach here, fail with a helpful error
    env_path = getattr(env, "path", lambda: "<?>")()
    raise KeyError(
        f"LMDB missing key '{array_name}_shape' in {env_path}. "
        f"Available shape keys (first 50): {shapes_available or 'NONE'}. "
        f"Check that array_name='{array_name}' matches the writer’s prefix and that you pointed to the right LMDB."
    )

 ### end of debug decode image shape ####
def create_image_grid(args, images_array, captions=None):
    # Set the dimensions of each individual image
    thumbnail_width = args.image_resolution
    thumbnail_height = args.image_resolution 

    # Spacing and margins
    caption_height = 30
    spacing = 15
    images_per_row = int(len(images_array) ** (1/2))  

    # Calculate grid dimensions
    total_width = (thumbnail_width + spacing) * images_per_row
    total_height = (thumbnail_height + caption_height + spacing) * (len(images_array) // images_per_row)

    # Create the big grid image with white background
    grid_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)

    # Load a font for the captions
    font = ImageFont.load_default()

    # Populate the grid with images and captions
    if captions is None:
        captions = ["" for _ in range(len(images_array))]

    for i, (img_data, caption) in enumerate(zip(images_array, captions)):
        img = Image.fromarray(img_data)
        img.thumbnail((thumbnail_width, thumbnail_height))

        # Calculate position in the grid
        x = (i % images_per_row) * (thumbnail_width + spacing)
        y = (i // images_per_row) * (thumbnail_height + caption_height + spacing)

        # Paste image and draw caption
        grid_img.paste(img, (x, y))

        wrapped_caption = textwrap.fill(str(caption), width=80)

        draw.text((x, y + thumbnail_height), f"{i:05d}_{wrapped_caption}", font=font, fill=(0, 0, 0))

    return grid_img 

class SDTextDataset(Dataset):
    def __init__(self, anno_path, tokenizer_one, is_sdxl=False, tokenizer_two=None):
        if anno_path.endswith(".txt"):
            self.all_prompts = []
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue 
                    else:
                        self.all_prompts.append(line)
        else:
            self.all_prompts = pickle.load(open(anno_path, "rb"))
    
        self.all_indices = list(range(len(self.all_prompts)))

        self.is_sdxl = is_sdxl # sdxl uses two tokenizers
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two

        print(f"Loaded {len(self.all_prompts)} prompts")

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx] or ""

        ids1 = self.tokenizer_one(
            [prompt], padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids  # [1, L]

        out = {
            'index': self.all_indices[idx],
            'key': prompt,
            'text_input_ids_one': ids1,       # trainer squeezes later
            'raw_prompt': prompt,             # <-- needed by Trainer
        }

        if self.is_sdxl:
            ids2 = self.tokenizer_two(
                [prompt], padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids  # [1, L2]
            out['text_input_ids_two'] = ids2
        return out
    
def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample

class NoOpContext:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 1)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def extract_text_embeddings(batch, accelerator, text_encoder_one, text_encoder_two):
    t1 = batch['text_input_ids_one']
    t2 = batch['text_input_ids_two']

    # Make shapes [B, L*]
    if t1.dim() == 3 and t1.size(1) == 1:  # [B,1,L1] -> [B,L1]
        t1 = t1.squeeze(1)
    if t2.dim() == 3 and t2.size(1) == 1:  # [B,1,L2] -> [B,L2]
        t2 = t2.squeeze(1)

    t1 = t1.to(accelerator.device)
    t2 = t2.to(accelerator.device)

    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for ids, enc in ((t1, text_encoder_one), (t2, text_encoder_two)):
        out = enc(ids.to(enc.device), output_hidden_states=True)
        pooled_prompt_embeds = out[0]  # last pooled (second encoder will overwrite, as intended)
        hs = out.hidden_states[-2]     # [B, L, D]
        prompt_embeds_list.append(hs)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)  # [B, L, D1+D2]
    pooled_prompt_embeds = pooled_prompt_embeds.view(len(t1), -1)
    return prompt_embeds, pooled_prompt_embeds


