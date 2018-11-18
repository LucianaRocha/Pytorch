model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        loss_2 = 0
        accuracy = 0        
        ## TODO: Implement the validation pass and print out the validation accuracy
        
        with torch.no_grad():
            for images, labels in testloader:
                log_ps_2 = model(images)
                loss_2 += criterion(log_ps_2, labels)
                
                ps_2 = torch.exp(log_ps_2)
                top_p, top_class = ps_2.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(loss_2/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(loss_2/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))