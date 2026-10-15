function xoverKids = permutationCrossover(parents,options,GenomeLength,~,~,thisPopulation)

nKids = length(parents)/2;
xoverKids = zeros(nKids,GenomeLength);

index = 1;

for i=1:nKids
    parent1 = thisPopulation(parents(index),:);
    index = index + 1;
    parent2 = thisPopulation(parents(index),:);
    index = index + 1;

    cutPoint = randi(length(parent1), 1);
    head = parent1(1, 1:cutPoint);
    indexes = not(ismember(parent2, head));
    tail = parent2(1, indexes);

    xoverKids(i,:) = [ head, tail ];

end
