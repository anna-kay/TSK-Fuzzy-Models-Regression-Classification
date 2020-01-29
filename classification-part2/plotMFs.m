%% Plot Membership Functions of FIS

function  plotMFs(fis,num_in)

    for i=1:num_in
        subplot(10,2,i);
        plotmf(fis,'input',i); grid on;
    end

end