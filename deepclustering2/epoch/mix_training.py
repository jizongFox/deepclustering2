from torch.cuda.amp import GradScaler

scaler = None


def scale_loss(loss, optimizer):
    global scaler
    if scaler is None:
        scaler = GradScaler()
    new_loss = scaler.scale(loss)
    yield new_loss
    scaler.step(optimizer)
    scaler.update()
