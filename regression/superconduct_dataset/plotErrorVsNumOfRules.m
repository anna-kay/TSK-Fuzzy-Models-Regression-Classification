tenFeatures = gridSearchData(43:48, 3:4);

plot(tenFeatures(:, 2), tenFeatures(:, 1) , '--o')
xlabel('# of rules')
ylabel('Error')
title('Error/Number of rules, using 10 features')