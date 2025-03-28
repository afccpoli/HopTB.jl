module Hall

using LinearAlgebra, Distributed
using ..HopTB
using ..HopTB.Utilities: fermidirac, constructmeshkpts, splitkpts
using ..HopTB.Parallel: ParallelFunction, claim!, stop!, parallel_sum
using ProgressMeter
using Distributed

export getahc


function _getahc(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64};
    Ts::Vector{Float64} = [0.0], μs::Vector{Float64} = [0.0])
    nkpts = size(kpts, 2)
    itgrd = zeros(ComplexF64, length(Ts), length(μs))
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, egvecs = geteig(atm, k)
        order = [0, 0, 0]
        order[α] = 1
        Sbar_α = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        order = [0, 0, 0]
        order[β] = 1
        Sbar_β = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        Abar_α = egvecs' * HopTB.getAw(atm, α, k) * egvecs
        Abar_β = egvecs' * HopTB.getAw(atm, β, k) * egvecs
        Dα = HopTB.getD(atm, α, k)
        Dβ = HopTB.getD(atm, β, k)
        order = [0, 0, 0]
        order[α] = 1
        dAw_βα = HopTB.getdAw(atm, β, Tuple(order), k)
        order = [0, 0, 0]
        order[β] = 1
        dAw_αβ = HopTB.getdAw(atm, α, Tuple(order), k)
        Ωbar_αβ = egvecs' * (dAw_βα - dAw_αβ) * egvecs
        tmp1 = Sbar_α*Abar_β
        tmp2 = Sbar_β*Abar_α
        for iT in 1:length(Ts)
            for iμ in 1:length(μs)
                for n in 1:atm.norbits
                    f = fermidirac(Ts[iT], egvals[n]-μs[iμ])
                    itgrd[iT, iμ] +=  f*(Ωbar_αβ[n, n]-tmp1[n, n]+tmp2[n, n])
                end
                for n in 1:atm.norbits, m in 1:atm.norbits
                    fm = fermidirac(Ts[iT], egvals[m] - μs[iμ])
                    fn = fermidirac(Ts[iT], egvals[n] - μs[iμ])
                    itgrd[iT, iμ] += (fm - fn) * (im * Dα[n, m] * Dβ[m, n] + Dα[n, m] * Abar_β[m, n] - Dβ[n, m] * Abar_α[m, n])
                end
            end
        end
    end
    return real.(itgrd)
end


@doc raw"""
```julia
getahc(atm::AbstractTBModel, α::Int64, β::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0])::Matrix{Float64}
```

Calculate anomalous Hall conductivity $σ^{αβ}$.

Anomalous Hall conductivity is defined by
```math
σ^{αβ}=-\frac{e^2}{ħ}\int\frac{d\boldsymbol{k}}{(2pi)^3}f_nΩ_{nn}^{αβ}.
```

The returned matrix $σ^{αβ}[m, n]$ is AHC for temperature Ts[m] and
chemical potential μs[n].

The returned AHC is in unit (Ω⋅cm)^-1.
"""
function getahc(atm::AbstractTBModel, α::Int64, β::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64} = [0.0], μs::Vector{Float64} = [0.0])
    @assert size(nkmesh, 1) == 3
    nkpts = prod(nkmesh)
    kpts = HopTB.Utilities.constructmeshkpts(nkmesh)
    kptslist = HopTB.Utilities.splitkpts(kpts, nworkers())

    jobs = Vector{Future}()
    for iw in 1:nworkers()
        job = @spawn _getahc(atm, α, β, kptslist[iw]; Ts = Ts, μs = μs)
        append!(jobs, [job])
    end


    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += HopTB.Utilities.safe_fetch(jobs[iw])
    end

    bzvol = abs(dot(cross(atm.rlat[:, 1], atm.rlat[:, 2]), atm.rlat[:, 3]))
    return σs * bzvol / nkpts * (-98.130728142) # -e**2/(hbar*(2pi)^3)*1.0e10/100
end


function _collect_berry_curvature(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64})
    nkpts = size(kpts, 2)
    berry_curvature = zeros(atm.norbits, nkpts)
    for ik in 1:nkpts
        k = kpts[:, ik]
        egvals, egvecs = geteig(atm, k)
        order = [0, 0, 0]; order[α] = 1; Sbar_α = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        order = [0, 0, 0]; order[β] = 1; Sbar_β = egvecs' * getdS(atm, Tuple(order), k) * egvecs
        Abar_α = egvecs' * HopTB.getAw(atm, α, k) * egvecs
        Abar_β = egvecs' * HopTB.getAw(atm, β, k) * egvecs
        Dα = HopTB.getD(atm, α, k)
        Dβ = HopTB.getD(atm, β, k)
        order = [0, 0, 0]; order[α] = 1; dAw_βα = HopTB.getdAw(atm, β, Tuple(order), k)
        order = [0, 0, 0]; order[β] = 1; dAw_αβ = HopTB.getdAw(atm, α, Tuple(order), k)
        Ωbar_αβ = egvecs' * (dAw_βα - dAw_αβ) * egvecs
        berry_curvature[:, ik] = real.(diag(Ωbar_αβ - Sbar_α * Abar_β + Sbar_β * Abar_α - im * Dα * Dβ + 
            im * Dβ * Dα - Dα * Abar_β + Abar_β * Dα + Dβ * Abar_α - Abar_α * Dβ))
    end
    return berry_curvature
end

"""
```julia
collect_berry_curvature(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64})::Matrix{Float64}
```

Collect berry curvature.

Standard units is used (eV and Å).

The returned matrix Ω[n, ik] is berry curvature for band n at ik point.
"""
function collect_berry_curvature(atm::AbstractTBModel, α::Int64, β::Int64, kpts::AbstractMatrix{Float64})
    nkpts = size(kpts, 2)

    kptslist = HopTB.Utilities.splitkpts(kpts, nworkers())
    jobs = Vector{Future}()
    for iw in 1:nworkers()
        job = @spawn _collect_berry_curvature(atm, α, β, kptslist[iw])
        append!(jobs, [job])
    end

    result = zeros((atm.norbits, 0))
    for iw in 1:nworkers()
        result = cat(result, HopTB.Utilities.safe_fetch(jobs[iw]), dims = (2,))
    end
    return result
end


function shc_worker(kpts::Matrix{Float64}, tm::TBModel, α::Int64, β::Int64, γ::Int64,
    Ts::Vector{Float64}, μs::Vector{Float64}, ϵ::Float64)
    result = zeros(length(Ts), length(μs))
    nkpts = size(kpts, 2)
    for ik in 1:nkpts
        k = kpts[:, ik]
        vα = getvelocity(tm, α, k); vβ = getvelocity(tm, β, k)
        sγ = getspin(tm, γ, k)
        jαγ = (sγ*vα+vα*sγ)/2
        egvals, _ = geteig(tm, k)
        for (iT, T) in enumerate(Ts), (iμ, μ) in enumerate(μs)
            for n in 1:tm.norbits
                ϵn = egvals[n]
                fn = fermidirac(T, ϵn-μ)
                for m in 1:tm.norbits
                    ϵm = egvals[m]
                    result[iT, iμ] += -fn*imag(jαγ[n, m]*vβ[m, n]/((ϵn-ϵm)^2+ϵ^2))
                end
            end
        end
    end
    return result
end

@doc raw"""
```julia
getshc(tm::TBModel, α::Int64, β::Int64, γ::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0], ϵ::Float64=0.1)::Matrix{Float64}
```

Calculate spin Hall conductivity for different temperature (`Ts`, first dimension)
and chemical potential (`μs`, second dimension).

Spin Hall conductivity is defined as
```math
σ_{αβ}^{γ} = eħ\int\frac{d^3 \boldsymbol{k}}{(2π)^3}\sum_n f_n Ω^{γ}_{n,αβ},
```
where the spin Berry curvature is
```math
Ω_{n,αβ}^{γ} = -2 \text{Im} [\sum_{m≠n} \frac{⟨n|\hat{j}_α^γ|m⟩⟨m|\hat{v}_β|n⟩}{(ϵ_n-ϵ_m)^2+ϵ^2}]
```
and the spin current operator is
```math
\hat{j}_α^γ = \frac{1}{2} \{\hat{v}_a, \hat{s}_c\}.
```

Spin Hall conductivity from this function is in ħ/e (Ω*cm)^-1.
"""
function getshc(tm::TBModel, α::Int64, β::Int64, γ::Int64, nkmesh::Vector{Int64};
    Ts::Vector{Float64}=[0.0], μs::Vector{Float64}=[0.0], ϵ::Float64=0.1)
    size(nkmesh, 1) == 3 || error("nkmesh should be a 3-element vector.")
    nkpts = prod(nkmesh)
    kptslist = splitkpts(constructmeshkpts(nkmesh), nworkers())
    pf = ParallelFunction(shc_worker, tm, α, β, γ, Ts, μs, ϵ)

    for iw in 1:nworkers()
        pf(kptslist[iw])
    end
    σs = zeros(Float64, length(Ts), length(μs))
    for iw in 1:nworkers()
        σs += claim!(pf)
    end
    stop!(pf)
    bzvol = abs((tm.rlat[:, 1]×tm.rlat[:, 2])⋅tm.rlat[:, 3])
    return σs*bzvol/nkpts*98.130728142 # e**2/(hbar*(2π)^3)*1.0e10/100
end


################################################################################
##  Berry curvature dipole
################################################################################


@doc raw"""
```
get_berry_curvature_dipole(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    fss::Vector{FermiSurface}
)::Float64
```

Calculate
```math
\sum_n \int_{\text{FS}_n} \frac{d \sigma}{(2\pi)^3} \Omega_{n}^{\alpha \beta} \frac{v_n^{\gamma}}{|\boldsymbol{v}_n|}
```
which is related to the Berry curvature dipole contribution to the second order photocurrent.
"""
function get_berry_curvature_dipole(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    fss::Vector{FermiSurface}
)
    result = 0.0
    for fs in fss
        result += parallel_sum(ik -> begin
            k = fs.ks[:, ik]
            Ω = get_berry_curvature(tm, α, β, k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            Ω * v[γ] * fs.weights[ik] / norm(v)
        end, 1:size(fs.ks, 2), 0.0)
    end
    return result / (2π)^3
end




################################################################################
##  Berry curvature quadrupole
################################################################################

@doc raw"""
```
get_berry_curvature_quadrupole(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64
    fss::Vector{FermiSurface}
)::Float64
```

Calculate
```math
\sum_n \int_{\text{FS}_n} \frac{d \sigma}{(2\pi)^3} \Omega_{n}^{\alpha \beta} \frac{dk_{γ}d_{δ}Es_n}{|\boldsymbol{v}_n|}
```
which is related to the Berry curvature quadrupole contribution to the second order photocurrent.
"""
function get_berry_curvature_quadrupole(
    tm::AbstractTBModel, 
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64,
    fss::Vector{FermiSurface}
)
    result = 0.0
    for fs in fss
        result += parallel_sum(ik -> begin
            k = fs.ks[:,ik]
            Ω= get_berry_curvature(tm, α, β, k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            Ω * v[δ]* fs.weights[ik] / norm(v)
        end, 1:size(fs.ks, 2), 0.0)
    end
    return result / (2π)^3
end


@doc raw"""
```
get_berry_curvature_quadrupole(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64
    fss::Vector{FermiSurface}
)::Float64
```

Calculate
```math
\sum_n \int_{\text{FS}_n} \frac{d \sigma}{(2\pi)^3} \Omega_{n}^{\alpha \beta} \frac{dk_{γ}d_{δ}Es_n}{|\boldsymbol{v}_n|}
```
which is related to the Berry curvature quadrupole contribution to the second order photocurrent.
"""
function get_berry_curvature_quadrupoledft(
    tm::AbstractTBModel, 
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64,
    fss::Vector{FermiSurface}
)
    result = 0.0
    for fs in fss
        result += parallel_sum(ik -> begin
            k = fs.ks[:,ik]
            Ω= HopTB.get_berry_curvature_dipoledft(tm, α, β, γ, k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            Ω * v[δ] * fs.weights[ik] / norm(v)
        end, 1:size(fs.ks, 2), 0.0)
    end
    return result / (2π)^3
end


function get_quantum_metric_quadrupoledft(
    tm::AbstractTBModel, 
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64,
    fss::Vector{FermiSurface}
)
    result = 0.0
    for fs in fss
        result += parallel_sum(ik -> begin
            k = fs.ks[:,ik]
            Ω= HopTB.get_quantum_metric_dipoledft(tm, α, β, γ, k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            Ω * v[δ] * fs.weights[ik] / norm(v)
        end, 1:size(fs.ks, 2), 0.0)
    end
    return result / (2π)^3
end






function get_berry_curvature_quadrupoledft3(
    tm::AbstractTBModel, 
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64,
    fss::Vector{FermiSurface};
    verbose::Bool=true,
    save_details::Bool=false
)
    total_result = 0.0
    details = save_details ? Vector{Tuple{Int,Float64,Float64}}() : nothing
    
    total_points = sum(fs -> size(fs.ks, 2), fss)
    prog = verbose ? Progress(total_points, 0.5, "Computing Berry Curvature...") : nothing
    
    print_lock = ReentrantLock()
    
    for (fs_idx, fs) in enumerate(fss)
        fs_result = parallel_sum(ik -> begin
            k = fs.ks[:,ik]
            Ω = HopTB.get_berry_curvature_dipoledft(tm, α, β, γ, k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            v_norm = norm(v)
            contribution = Ω * v[δ] * fs.weights[ik] / v_norm
            
            if verbose
                lock(print_lock) do
                    # 使用字符串插值代替 @printf
                    msg = "[FS $(lpad(fs_idx,2))][k-point $(rpad(ik,5))] Contribution = $(signbit(contribution) ? "-" : "+")$(abs(contribution))"
                    println(msg)
                    next!(prog)
                end
            end
            
            save_details && push!(details, (fs_idx, ik, contribution))
            contribution
        end, 1:size(fs.ks, 2), 0.0)
        
        verbose && @info "Fermi Surface $(fs_idx) integral: $(fs_result)"
        total_result += fs_result
    end
    
    final_result = total_result / (2π)^3
    verbose && @info "Final result: $(final_result) (normalized by 1/(2π)^3)"
    return save_details ? (final_result, details) : final_result
end



@doc raw"""
```
get_third_Drude-like_term(
    tm::AbstractTBModel,
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64
    fss::Vector{FermiSurface}
)::Float64
```

Calculate
```math
\sum_n \int_{\text{FS}_n} \frac{d \sigma}{(2\pi)^3} \Omega_{n}^{\alpha \beta} \frac{dk_{γ}d_{δ}Es_n}{|\boldsymbol{v}_n|}
```
which is related to the Berry curvature quadrupole contribution to the second order photocurrent.
"""
function get_third_Drudelike_term(
    tm::AbstractTBModel, 
    α::Int64,
    β::Int64,
    γ::Int64,
    δ::Int64,
    fss::Vector{FermiSurface}
)
    result = 0.0
    for fs in fss
        result += parallel_sum(ik -> begin
            k = fs.ks[:,ik]
            Eabc=getdEs(tm,β,γ,δ,k)[fs.bandidx]
            v = real([getvelocity(tm, i, k)[fs.bandidx, fs.bandidx] for i in 1:3])
            v[α] * Eabc * fs.weights[ik] / norm(v)
        end, 1:size(fs.ks, 2), 0.0)
    end
    return result / (2π)^3
end
end