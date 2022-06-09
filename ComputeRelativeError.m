function errors = ComputeRelativeError(GradsAn, GradsNum)

    errors = [];

    for f = fieldnames(GradsAn)'
        errors(end + 1) = max( abs(GradsNum.(f{1}) - GradsAn.(f{1})) ./ max(abs(GradsNum.(f{1})) + abs(GradsAn.(f{1})), eps), [], 'all' );
    end

end