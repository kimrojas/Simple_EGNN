import torch
import math
import time
from tqdm import tqdm

bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
textsize = 14
default_dtype = torch.float64


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def evaluate(model, dataloader, loss_fn, loss_fn_mae, device):
    model.eval()
    loss_cumulative = 0.0
    loss_cumulative_mae = 0.0
    start_time = time.time()
    with torch.no_grad():
        for j, d in enumerate(dataloader):
            d.to(device)
            output, _ = model(d)
            loss = loss_fn(output, d.energy).cpu()
            loss_mae = loss_fn_mae(output, d.energy).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()
    return loss_cumulative / len(dataloader), loss_cumulative_mae / len(dataloader)


def train(
    model,
    optimizer,
    dataloader_train,
    dataloader_valid,
    loss_fn,
    loss_fn_mae,
    run_name,
    max_iter=101,
    scheduler=None,
    device="cpu",
):
    model.to(device)

    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()

    try:
        model.load_state_dict(torch.load(run_name + ".torch")["state"])
    except:
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + ".torch")
        history = results["history"]
        s0 = history[-1]["step"] + 1

    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.0
        loss_cumulative_mae = 0.0

        for j, d in tqdm(enumerate(dataloader_train), total=len(dataloader_train), bar_format=bar_format):
            d.to(device)
            output, _ = model(d)
            loss = loss_fn(output, d.energy).cpu()
            loss_mae = loss_fn_mae(output, d.energy).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
            loss_cumulative_mae = loss_cumulative_mae + loss_mae.detach().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        wall = end_time - start_time

        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, device)
            train_avg_loss = evaluate(model, dataloader_train, loss_fn, loss_fn_mae, device)

            history.append(
                {
                    "step": s0 + step,
                    "wall": wall,
                    "batch": {
                        "loss": loss.item(),
                        "mean_abs": loss_mae.item(),
                    },
                    "valid": {
                        "loss": valid_avg_loss[0],
                        "mean_abs": valid_avg_loss[1],
                    },
                    "train": {
                        "loss": train_avg_loss[0],
                        "mean_abs": train_avg_loss[1],
                    },
                }
            )

            results = {"history": history, "state": model.state_dict()}

            print(
                f"Iteration {step+1:4d}   "
                + f"train loss = {train_avg_loss[0]:8.4f}   "
                + f"valid loss = {valid_avg_loss[0]:8.4f}   "
                + f"elapsed time = {time.strftime('%H:%M:%S', time.gmtime(wall))}"
            )

            with open(run_name + ".torch", "wb") as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step()
