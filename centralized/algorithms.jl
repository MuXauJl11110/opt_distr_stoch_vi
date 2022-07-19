using Distributions:length
using Base:throw_eachindex_mismatch_indices, setindex
using Distributions
using StatsBase
include("utils.jl")


# We assume here that proj is implemented symmetrically for primal and dual variables
function extraGrad(A::Array{Float64,2}, proj::Function, z0::Array{Float64,1},  τ::Float64, iter::Int64; tol=1e-6)
    m, n = size(A)
    @assert  size(z0)[1] == m + n
    energy = Float64[]
    x, y = z0[1:n], z0[(n + 1):end]
    iter = ceil(Int, iter / 2) # since every iteration is 2 epochs.

    for i in 1:iter
        x, y, energy = extraGradUpdate(A, proj, x, y, energy, τ)
        if energy[end] < tol
            println("The extragradient algorithm achieves $tol accuracy in $(i - 1) iterations")
            break
        end
    end
    return energy, x, y
end

"""
    extraGradUpdate: main update for extragradient method for matrix games on a simplex.
"""
function extraGradUpdate(A, proj, x, y, energy, τ)

    Ax, ATy =  A * x, A' * y
    gap = maximum(Ax) - minimum(ATy)
    append!(energy, gap)
    x_ = proj(x .- τ .* ATy)
    y_ = proj(y .+ τ .* Ax)
    # AA: added below to avoid multiplying by 2 when plotting
    Ax, ATy =  A * x_, A' * y_
    gap = maximum(Ax) - minimum(ATy)
    append!(energy, gap)
    x = proj(x .- τ .* ATy)
    y = proj(y .+ τ .* Ax)
    return x, y, energy
end

######################### ExtraGrad ######################################################
"""
    stochastic ExtraGrad with variance reduction, loopless variant.

distr = true: uses weighted sampling. It uses a sampling |A[i,:]|^2/|A|_F and similarly for columns.
"""

function stochExtraGradLoopless(A::Array{Float64,2}, proj::Function, z0::Array{Float64,1},  τ::Float64,
                           α::Float64, p::Float64, b::Int64, max_epoch::Int64; distr=false, tol=1e-6)
    m, n = size(A)
    @assert m + n == size(z0)[1]
    if b > m + n || b < 1
        throw(ArgumentError("Incorrect batch size: $b!"))
    end
    iter = ceil(Int, max_epoch * 2 * m * n / (p * 2 * m * n + m + n ))
    cheap_update = (m + n) / (2 * m * n)
    update_wArray = rand(Bernoulli(p), iter)

    arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, iter, b, frobenius=distr)
    
    x, y = z0[1:n], z0[(n + 1):end]
    wx, wy = x, y
    energy = [maximum(A * wx) - minimum(A' * wy)]
    epoch = [0.0]
    epoch_count = 0.0

    for k in 1:iter
        update_w = update_wArray[k]
        i, j = arrayI[:, k], arrayJ[:, k]
        x, y, wx, wy, energy, epoch, epoch_count =
        stochExtraGradLooplessBatch_update(A, proj, x, y, wx, wy, energy, epoch, epoch_count, α, τ, 
        b, i, j, cheap_update, update_w, rows_weights, columns_weights)
        if energy[end] < tol
            total_epoch = ceil(sum(epoch))
            println("StocExtraGrad-VR achieves $tol accuracy in $(total_epoch) epochs")
            break
        end
    end
    gap = maximum(A * x) - minimum(A' * y)
    append!(energy, gap)
    append!(epoch, 1.0)
    return energy, x, y, cumsum(epoch)
end

function stochExtraGradLooplessBatch_update(A, proj, x, y, wx, wy, energy, epoch, epoch_count, α, τ, b, 
    i, j, cheap_update, update_w, rows_weights, columns_weights)

    m, n = size(A)
    
    x_ = α .* x .+ (1 - α) .* wx
    y_ = α .* y .+ (1 - α) .* wy
    xx = proj(x_ .- τ .* (A' * wy))
    yy = proj(y_ .- τ .* (-A * wx))
    xx_, yy_ = zeros(n), zeros(m)
    setindex!(xx_, (xx - wx)[j], j)
    setindex!(yy_, (yy - wy)[i], i)
    x = proj(x_ .- τ .* ((A' * wy) .+ 1 / b .* (A' * (yy_ ./ rows_weights))))
    y = proj(y_ .- τ .* ((-A * wx) .+ 1 / b .* (-A * (xx_ ./ columns_weights))))
    epoch_count += cheap_update * b
    if update_w
        wx, wy = x, y
        gap = maximum(A * wx) - minimum(A' * wy)
        append!(energy, gap)
        epoch_count += 1.0
        append!(epoch, epoch_count)
        epoch_count = 0.0
    end

    return x, y, wx, wy, energy, epoch, epoch_count
end

"""
    stochastic ExtraGrad with variance reduction, looped variant.

distr = true: uses weighted sampling. It uses a sampling |A[i,:]|^2/|A|_F and similarly for columns.
"""

function stochExtraGradLooped(A::Array{Float64,2}, proj::Function, z0::Array{Float64,1},  τ::Float64,
    α::Float64, T::Int64, b::Int64, max_epoch::Int64; distr=false, tol=1e-6)
    m, n = size(A)
    @assert m + n == size(z0)[1]
    outer_iter = ceil(Int, max_epoch * 2 * m * n / (2 * m * n + T * (m + n)))
    cheap_update = (m + n) / (2 * m * n)
    cost_per_epoch = 1 + T * cheap_update * b

    arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, T * outer_iter, b, frobenius=distr)

    x, y = z0[1:n], z0[(n + 1):end]
    wx, wy = x, y
    energy = [maximum(A * x) - minimum(A' * y)]
    x_avg, y_avg = zeros(Float64, n), zeros(Float64, m)

    for k in 1:outer_iter
        x, y, x_avg, y_avg =
        stochExtraGradLooped_update(A, proj, x, y, x_avg, y_avg, wx, wy, α, τ, T, b, arrayI, arrayJ, 
        k, rows_weights, columns_weights)

        append!(energy, maximum(A * x_avg) - minimum(A' * y_avg))    
        wx, wy = x_avg, y_avg
        if energy[end] < tol
            println("StocExtraGrad-VR-Looped achieved $tol accuracy")
            break
        end
    end
    # x_avg is on the primal space, so no need to softmax here.
    running_cost = Array(1:length(energy)) * cost_per_epoch
    return energy, x, y, running_cost
end

function stochExtraGradLooped_update(A, proj, x, y, x_avg, y_avg, wx, wy, α, τ, T, b, arrayI, arrayJ, 
    k, rows_weights, columns_weights)
    m, n = size(A)

    for t in 1:T
        i = arrayI[:, (k - 1) * T + t]
        j = arrayJ[:, (k - 1) * T + t]
        
        x_ = α .* x .+ (1 - α) .* wx
        y_ = α .* y .+ (1 - α) .* wy
        xx = proj(x_ .- τ .* (A' * wy))
        yy = proj(y_ .- τ .* (-A * wx))
        xx_, yy_ = zeros(n), zeros(m)
        setindex!(xx_, (xx - wx)[j], j)
        setindex!(yy_, (yy - wy)[i], i)
        x = proj(x_ .- τ .* (A' * (wy .+ 1 / b .* (yy_ ./ rows_weights))))
        y = proj(y_ .- τ .* (-A * (wx .+ 1 / b .* (xx_ ./ columns_weights))))

        x_avg = 1 / t .* x .+ (1 - 1 / t) .* x_avg
        y_avg = 1 / t .* y .+ (1 - 1 / t) .* y_avg
    end

    return x, y, x_avg, y_avg
end


######################### Algorithm 1 ####################################################
"""
    stochastic Algorithm 1 with variance reduction, loopless variant.

distr = true: uses weighted sampling. It uses a sampling |A[i,:]|^2/|A|_F and similarly for columns.
"""

function stochAlgorithm1(A::Array{Float64,2}, proj::Function,
    z0::Array{Float64,1}, w0::Array{Float64,1}, α::Float64, γ::Float64, η::Float64, p::Float64, b::Int64,
    max_epoch::Int64; distr=false, tol=1e-6)

    m, n = size(A)
    @assert m + n == size(z0)[1]
    iter = ceil(Int, max_epoch * 2 * m * n / (p * 2 * m * n + m + n))
    if b > m + n || b < 1
        throw(ArgumentError("Incorrect batch size: $b!"))
    end
    cheap_update = (m + n) / (2 * m * n)
    update_wArray = rand(Bernoulli(p), iter)

    arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, iter, b, frobenius=distr)
    
    x, y = z0[1:n], z0[(n + 1):end]
    x_old, y_old = x, y
    wx, wy = w0[1:n], w0[(n + 1):end]
    wx_old, wy_old = wx, wy
    energy = [maximum(A * wx) - minimum(A' * wy)]
    epoch = [0.0]
    epoch_count = 0.0

    for k in 1:iter
        update_w = update_wArray[k]
        i, j = arrayI[:, k], arrayJ[:, k]
        x, y, x_old, y_old, wx, wy, wx_old, wy_old, energy, epoch, epoch_count =
        stochAlgorithm1Batch_update(A, proj, x, y, x_old, y_old, wx, wy, wx_old, wy_old,
                                        energy, epoch, epoch_count, α, γ, η, b, i, j, cheap_update,
                                        update_w, rows_weights, columns_weights)
        if energy[end] < tol
            total_epoch = ceil(sum(epoch))
            println("stochAlgorithm1 achieves $tol accuracy in $(total_epoch) epochs")
            break
        end
    end
    gap = maximum(A * x) - minimum(A' * y)
    append!(energy, gap)
    append!(epoch, 1.0)
    return energy, x, y, cumsum(epoch)
end


function stochAlgorithm1Batch_update(A, proj, x, y, x_old, y_old, wx, wy, wx_old, wy_old, energy, epoch,
        epoch_count, α, γ, η, b, i, j, cheap_update, update_w, rows_weights, columns_weights)
    
    x_, y_ = zeros(n), zeros(m)
    setindex!(x_, (y - wy_old + α * (y - y_old))[j], j)
    setindex!(y_, (x - wx_old + α * (x - x_old))[i], i)    
    
    delta_x = 1 / b .* (A' * (y_ ./ rows_weights)) .+ A' * wy_old
    delta_y = 1 / b .* (-A * (x_ ./ columns_weights)) .- A * wx_old
    x = proj(x .+ γ .* (wx .- x) .- η .* delta_x)
    y = proj(y .+ γ .* (wx .- x) .- η .* delta_y)

    epoch_count += cheap_update * b
    wx_old, wy_old = wx, wy
    x_old, y_old = x, y
    if update_w
        wx, wy = x, y
        gap = maximum(A * wx) - minimum(A' * wy)
        append!(energy, gap)
        epoch_count += 1.0
        append!(epoch, epoch_count)
        epoch_count = 0.0
    end

    return x, y, x_old, y_old, wx, wy, wx_old, wy_old, energy, epoch, epoch_count
end

"""
Carmon et al. variant

distr = true: uses weighted sampling. It uses a sampling |A[i,:]|^2/|A|_F and similarly for columns.
"""

function stochMPCarmon(A::Array{Float64,2}, proj::Function, z0::Array{Float64,1},  α::Float64,
                η::Float64, b::Int64, max_epoch::Int64; distr=false, tol=1e-6)

    m, n = size(A)
    @assert m + n == size(z0)[1]
    T = ceil(Int, 4 / (η * α))
    outer_iter = ceil(Int, max_epoch * 2 * m * n / (4 * m * n + T * (m + n)))
    cheap_update = (m + n) / (2 * m * n) * b
    # averaged cost for each time we save data. In [Carmon et al], every outer iteration
    # does two full updates, thus 2 times save data. Cost of one outer iteration
    # is (2 + K*cheap_update) so cost per save is (2 + K*cheap_update)/2
    cost_per_save = (2 + T * cheap_update) / 2

    x, y = z0[1:n], z0[(n + 1):end]
    
    x0, y0 = x, y
    Awx, Awy = A * x, A' * y
    arrayI, arrayJ, rows_weights, columns_weights = sample_with_Frobenius(A, T * outer_iter, b, frobenius=distr)
    energy = [maximum(Awx) - minimum(Awy)]

    x_avg, y_avg = zeros(Float64, n), zeros(Float64, m)
    for k in 1:outer_iter
        Awx, Awy, energy = compute_full_operator(A, x0, y0, energy)
        
        if energy[end] < tol
            break
        end
        x, y, x_avg, y_avg =
        stochMPCarmon_update_euclidean(A, proj, x_avg, y_avg, x0, y0, Awx, Awy, α, η, T, k, b, 
        arrayI, arrayJ, rows_weights, columns_weights)
        # x_avg is in the primal space, so no need to softmax here.
        Awx, Awy, energy = compute_full_operator(A, x_avg, y_avg, energy)

        if energy[end] < tol
            # total_epoch = ceil(sum(epoch))
            println("Stochastic Carmon algorithm achieved $tol accuracy")
            break
        end

        x0, y0 = stochMPCarmon_fullupdate_euclidean(proj, x0, y0, Awx, Awy, α)
    end
    running_cost = Array(0:length(energy) - 1) * cost_per_save
    return energy, x, y, running_cost
end


function stochMPCarmon_update_euclidean(A, proj, x_avg, y_avg, x0, y0, Awx, Awy, α, η, T, k, b, 
    arrayI, arrayJ, rows_weights, columns_weights)
    x, y = x0, y0
    m, n = size(A)
        
    for t in 1:T
        i = arrayI[:, (k - 1) * T + t]
        j = arrayJ[:, (k - 1) * T + t]

        x_, y_ = zeros(n), zeros(m)
        setindex!(x_, (x - x0)[j], j)
        setindex!(y_, (y - y0)[i], i)
        x = proj((x .+ (η * α / 2) .* x0 .- η .* (Awy .+ 1 / b .* A' * (y_ ./ rows_weights))) ./ (1 + η * α / 2))
        y = proj((y .+ (η * α / 2) .* y0 .- η .* (-Awx .- 1 / b .* A * (x_ ./ columns_weights))) ./ (1 + η * α / 2))
        x_avg = (1 / t) .* x .+ (1 - 1 / t) .* x_avg
        y_avg = (1 / t) .* y .+ (1 - 1 / t) .* y_avg
    end
    return x, y, x_avg, y_avg
end

function stochMPCarmon_fullupdate_euclidean(proj, x0, y0, Awx, Awy, α)
    x = proj(x0 .- (Awy ./ α))
    y = proj(y0 .+ (Awx ./ α))
    return x, y
end

function compute_full_operator(A, x_avg, y_avg, energy)
    Awx, Awy = A * x_avg, A' * y_avg
    append!(energy, maximum(Awx) - minimum(Awy))
    return Awx, Awy, energy
end

function sample_with_Frobenius(A, iter::Int64, b::Int64; frobenius=false)
    m, n = size(A)
    arrayI = Array{Int64}(undef, b, iter)
    arrayJ = Array{Int64}(undef, b, iter)
    if frobenius
        frobenius_norm = norm(A)
        rows_norm = [norm(A[i, :]) for i in 1:m]
        columns_norm = [norm(A[:, j]) for j in 1:n]
        rows_weights = rows_norm.^2 / frobenius_norm^2
        columns_weights = columns_norm.^2 / frobenius_norm^2
        for i in (1:iter)
            arrayI[:, i] = sample((1:m), Weights(rows_weights), b, replace=false)
            arrayJ[:, i] = sample((1:n), Weights(columns_weights), b, replace=false)
        end
        return arrayI, arrayJ, rows_weights, columns_weights
    else
        for i in (1:iter)
            arrayI[:, i] = sample((1:m), b, replace=false)
            arrayJ[:, i] = sample((1:n), b, replace=false)
        end
        return arrayI, arrayJ, ones(m), ones(n)
    end
end