from dataclasses import dataclass
from typing import Type, cast

from optimagic.optimization.algorithm import Algorithm
from optimagic.optimizers.bhhh import BHHH
from optimagic.optimizers.fides import Fides
from optimagic.optimizers.ipopt import Ipopt
from optimagic.optimizers.nag_optimizers import NagDFOLS, NagPyBOBYQA
from optimagic.optimizers.neldermead import NelderMeadParallel
from optimagic.optimizers.nlopt_optimizers import (
    NloptBOBYQA,
    NloptCCSAQ,
    NloptCOBYLA,
    NloptCRS2LM,
    NloptDirect,
    NloptESCH,
    NloptISRES,
    NloptLBFGSB,
    NloptMMA,
    NloptNelderMead,
    NloptNEWUOA,
    NloptPRAXIS,
    NloptSbplx,
    NloptSLSQP,
    NloptTNewton,
    NloptVAR,
)
from optimagic.optimizers.pounders import Pounders
from optimagic.optimizers.pygmo_optimizers import (
    PygmoBeeColony,
    PygmoCmaes,
    PygmoCompassSearch,
    PygmoDe,
    PygmoDe1220,
    PygmoGaco,
    PygmoGwo,
    PygmoIhs,
    PygmoMbh,
    PygmoPso,
    PygmoPsoGen,
    PygmoSade,
    PygmoSea,
    PygmoSga,
    PygmoSimulatedAnnealing,
    PygmoXnes,
)
from optimagic.optimizers.scipy_optimizers import (
    ScipyBasinhopping,
    ScipyBFGS,
    ScipyBrute,
    ScipyCOBYLA,
    ScipyConjugateGradient,
    ScipyDifferentialEvolution,
    ScipyDirect,
    ScipyDualAnnealing,
    ScipyLBFGSB,
    ScipyLSDogbox,
    ScipyLSLM,
    ScipyLSTRF,
    ScipyNelderMead,
    ScipyNewtonCG,
    ScipyPowell,
    ScipySHGO,
    ScipySLSQP,
    ScipyTruncatedNewton,
    ScipyTrustConstr,
)
from optimagic.optimizers.tao_optimizers import TAOPounders
from optimagic.optimizers.tranquilo import Tranquilo, TranquiloLS


@dataclass(frozen=True)
class AlgoSelection:
    @property
    def All(self) -> list[Type[Algorithm]]:
        raw = [field.default for field in self.__dataclass_fields__.values()]
        return cast(list[Type[Algorithm]], raw)

    @property
    def Available(self) -> list[Type[Algorithm]]:
        return [
            a
            for a in self.All
            if a.__algo_info__.is_available  # type: ignore
        ]


@dataclass(frozen=True)
class BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS


@dataclass(frozen=True)
class BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    scipy_shgo: Type[ScipySHGO] = ScipySHGO


@dataclass(frozen=True)
class BoundedGradientBasedLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr


@dataclass(frozen=True)
class BoundedGradientBasedLeastSquaresLocalAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF


@dataclass(frozen=True)
class BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Parallel(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientFreeParallelAlgorithms(AlgoSelection):
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Parallel(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Local(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def GradientFree(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientBasedAlgorithms(AlgoSelection):
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GlobalGradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class BoundedGradientBasedLocalAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def LeastSquares(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        return BoundedGradientBasedLeastSquaresLocalAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientBasedLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientBasedLeastSquaresLocalAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF

    @property
    def Bounded(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        return BoundedGradientBasedLeastSquaresLocalAlgorithms()


@dataclass(frozen=True)
class GradientBasedLikelihoodLocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH


@dataclass(frozen=True)
class BoundedGradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms()

    @property
    def Local(self) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class BoundedGradientBasedLeastSquaresAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF

    @property
    def Local(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        return BoundedGradientBasedLeastSquaresLocalAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientFreeAlgorithms(AlgoSelection):
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        return BoundedGlobalGradientFreeParallelAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeParallelAlgorithms(AlgoSelection):
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        return BoundedGlobalGradientFreeParallelAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeLocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        return BoundedGradientFreeLocalParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA

    @property
    def Bounded(self) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientFreeLeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        return GradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        return BoundedGradientFreeLocalParallelAlgorithms()

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        return GradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Local(self) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeLocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Local(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        return BoundedGlobalGradientFreeParallelAlgorithms()

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        return BoundedGradientFreeLocalParallelAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelAlgorithms()

    @property
    def Global(self) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        return GradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def GradientBased(self) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms()

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalParallelAlgorithms(AlgoSelection):
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        return BoundedGlobalGradientFreeParallelAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GlobalNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelAlgorithms()

    @property
    def GradientFree(self) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def GradientBased(self) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedLocalNonlinearConstrainedAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class BoundedLeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientBased(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        return BoundedGradientBasedLeastSquaresLocalAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        return BoundedLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        return BoundedGradientFreeLocalParallelAlgorithms()

    @property
    def LeastSquares(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        return BoundedLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class LeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        return BoundedLeastSquaresLocalParallelAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        return GradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        return BoundedLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class GlobalGradientBasedAlgorithms(AlgoSelection):
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalGradientBasedAlgorithms:
        return BoundedGlobalGradientBasedAlgorithms()

    @property
    def NonlinearConstrained(self) -> GlobalGradientBasedNonlinearConstrainedAlgorithms:
        return GlobalGradientBasedNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientBasedLocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedLocalAlgorithms:
        return BoundedGradientBasedLocalAlgorithms()

    @property
    def LeastSquares(self) -> GradientBasedLeastSquaresLocalAlgorithms:
        return GradientBasedLeastSquaresLocalAlgorithms()

    @property
    def Likelihood(self) -> GradientBasedLikelihoodLocalAlgorithms:
        return GradientBasedLikelihoodLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> GradientBasedLocalNonlinearConstrainedAlgorithms:
        return GradientBasedLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class BoundedGradientBasedAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalGradientBasedAlgorithms:
        return BoundedGlobalGradientBasedAlgorithms()

    @property
    def LeastSquares(self) -> BoundedGradientBasedLeastSquaresAlgorithms:
        return BoundedGradientBasedLeastSquaresAlgorithms()

    @property
    def Local(self) -> BoundedGradientBasedLocalAlgorithms:
        return BoundedGradientBasedLocalAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedNonlinearConstrainedAlgorithms()

    @property
    def Global(self) -> GlobalGradientBasedNonlinearConstrainedAlgorithms:
        return GlobalGradientBasedNonlinearConstrainedAlgorithms()

    @property
    def Local(self) -> GradientBasedLocalNonlinearConstrainedAlgorithms:
        return GradientBasedLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientBasedLeastSquaresAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF

    @property
    def Bounded(self) -> BoundedGradientBasedLeastSquaresAlgorithms:
        return BoundedGradientBasedLeastSquaresAlgorithms()

    @property
    def Local(self) -> GradientBasedLeastSquaresLocalAlgorithms:
        return GradientBasedLeastSquaresLocalAlgorithms()


@dataclass(frozen=True)
class GradientBasedLikelihoodAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH

    @property
    def Local(self) -> GradientBasedLikelihoodLocalAlgorithms:
        return GradientBasedLikelihoodLocalAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeAlgorithms(AlgoSelection):
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeAlgorithms:
        return BoundedGlobalGradientFreeAlgorithms()

    @property
    def NonlinearConstrained(self) -> GlobalGradientFreeNonlinearConstrainedAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GlobalGradientFreeParallelAlgorithms:
        return GlobalGradientFreeParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLocalAlgorithms:
        return BoundedGradientFreeLocalAlgorithms()

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresLocalAlgorithms:
        return GradientFreeLeastSquaresLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> GradientFreeLocalNonlinearConstrainedAlgorithms:
        return GradientFreeLocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GradientFreeLocalParallelAlgorithms:
        return GradientFreeLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalGradientFreeAlgorithms:
        return BoundedGlobalGradientFreeAlgorithms()

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresAlgorithms:
        return BoundedGradientFreeLeastSquaresAlgorithms()

    @property
    def Local(self) -> BoundedGradientFreeLocalAlgorithms:
        return BoundedGradientFreeLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> BoundedGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedGradientFreeParallelAlgorithms:
        return BoundedGradientFreeParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Global(self) -> GlobalGradientFreeNonlinearConstrainedAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Local(self) -> GradientFreeLocalNonlinearConstrainedAlgorithms:
        return GradientFreeLocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GradientFreeNonlinearConstrainedParallelAlgorithms:
        return GradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresAlgorithms:
        return BoundedGradientFreeLeastSquaresAlgorithms()

    @property
    def Local(self) -> GradientFreeLeastSquaresLocalAlgorithms:
        return GradientFreeLeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> GradientFreeLeastSquaresParallelAlgorithms:
        return GradientFreeLeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    pounders: Type[Pounders] = Pounders
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeParallelAlgorithms:
        return BoundedGradientFreeParallelAlgorithms()

    @property
    def Global(self) -> GlobalGradientFreeParallelAlgorithms:
        return GlobalGradientFreeParallelAlgorithms()

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresParallelAlgorithms:
        return GradientFreeLeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> GradientFreeLocalParallelAlgorithms:
        return GradientFreeLocalParallelAlgorithms()

    @property
    def NonlinearConstrained(
        self,
    ) -> GradientFreeNonlinearConstrainedParallelAlgorithms:
        return GradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalAlgorithms(AlgoSelection):
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def GradientBased(self) -> BoundedGlobalGradientBasedAlgorithms:
        return BoundedGlobalGradientBasedAlgorithms()

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeAlgorithms:
        return BoundedGlobalGradientFreeAlgorithms()

    @property
    def NonlinearConstrained(self) -> BoundedGlobalNonlinearConstrainedAlgorithms:
        return BoundedGlobalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedGlobalParallelAlgorithms:
        return BoundedGlobalParallelAlgorithms()


@dataclass(frozen=True)
class GlobalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalNonlinearConstrainedAlgorithms:
        return BoundedGlobalNonlinearConstrainedAlgorithms()

    @property
    def GradientBased(self) -> GlobalGradientBasedNonlinearConstrainedAlgorithms:
        return GlobalGradientBasedNonlinearConstrainedAlgorithms()

    @property
    def GradientFree(self) -> GlobalGradientFreeNonlinearConstrainedAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GlobalNonlinearConstrainedParallelAlgorithms:
        return GlobalNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class GlobalParallelAlgorithms(AlgoSelection):
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalParallelAlgorithms:
        return BoundedGlobalParallelAlgorithms()

    @property
    def GradientFree(self) -> GlobalGradientFreeParallelAlgorithms:
        return GlobalGradientFreeParallelAlgorithms()

    @property
    def NonlinearConstrained(self) -> GlobalNonlinearConstrainedParallelAlgorithms:
        return GlobalNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLocalAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientBased(self) -> BoundedGradientBasedLocalAlgorithms:
        return BoundedGradientBasedLocalAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalAlgorithms:
        return BoundedGradientFreeLocalAlgorithms()

    @property
    def LeastSquares(self) -> BoundedLeastSquaresLocalAlgorithms:
        return BoundedLeastSquaresLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> BoundedLocalNonlinearConstrainedAlgorithms:
        return BoundedLocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedLocalParallelAlgorithms:
        return BoundedLocalParallelAlgorithms()


@dataclass(frozen=True)
class LocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedLocalNonlinearConstrainedAlgorithms:
        return BoundedLocalNonlinearConstrainedAlgorithms()

    @property
    def GradientBased(self) -> GradientBasedLocalNonlinearConstrainedAlgorithms:
        return GradientBasedLocalNonlinearConstrainedAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLocalNonlinearConstrainedAlgorithms:
        return GradientFreeLocalNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class LeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresLocalAlgorithms:
        return BoundedLeastSquaresLocalAlgorithms()

    @property
    def GradientBased(self) -> GradientBasedLeastSquaresLocalAlgorithms:
        return GradientBasedLeastSquaresLocalAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresLocalAlgorithms:
        return GradientFreeLeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> LeastSquaresLocalParallelAlgorithms:
        return LeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class LikelihoodLocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH

    @property
    def GradientBased(self) -> GradientBasedLikelihoodLocalAlgorithms:
        return GradientBasedLikelihoodLocalAlgorithms()


@dataclass(frozen=True)
class LocalParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLocalParallelAlgorithms:
        return BoundedLocalParallelAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLocalParallelAlgorithms:
        return GradientFreeLocalParallelAlgorithms()

    @property
    def LeastSquares(self) -> LeastSquaresLocalParallelAlgorithms:
        return LeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalNonlinearConstrainedAlgorithms:
        return BoundedGlobalNonlinearConstrainedAlgorithms()

    @property
    def GradientBased(self) -> BoundedGradientBasedNonlinearConstrainedAlgorithms:
        return BoundedGradientBasedNonlinearConstrainedAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeNonlinearConstrainedAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Local(self) -> BoundedLocalNonlinearConstrainedAlgorithms:
        return BoundedLocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedNonlinearConstrainedParallelAlgorithms:
        return BoundedNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientBased(self) -> BoundedGradientBasedLeastSquaresAlgorithms:
        return BoundedGradientBasedLeastSquaresAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresAlgorithms:
        return BoundedGradientFreeLeastSquaresAlgorithms()

    @property
    def Local(self) -> BoundedLeastSquaresLocalAlgorithms:
        return BoundedLeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> BoundedLeastSquaresParallelAlgorithms:
        return BoundedLeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class BoundedParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalParallelAlgorithms:
        return BoundedGlobalParallelAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeParallelAlgorithms:
        return BoundedGradientFreeParallelAlgorithms()

    @property
    def LeastSquares(self) -> BoundedLeastSquaresParallelAlgorithms:
        return BoundedLeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> BoundedLocalParallelAlgorithms:
        return BoundedLocalParallelAlgorithms()

    @property
    def NonlinearConstrained(self) -> BoundedNonlinearConstrainedParallelAlgorithms:
        return BoundedNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class NonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedNonlinearConstrainedParallelAlgorithms:
        return BoundedNonlinearConstrainedParallelAlgorithms()

    @property
    def Global(self) -> GlobalNonlinearConstrainedParallelAlgorithms:
        return GlobalNonlinearConstrainedParallelAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeNonlinearConstrainedParallelAlgorithms:
        return GradientFreeNonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class LeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresParallelAlgorithms:
        return BoundedLeastSquaresParallelAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresParallelAlgorithms:
        return GradientFreeLeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> LeastSquaresLocalParallelAlgorithms:
        return LeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class GradientBasedAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedAlgorithms:
        return BoundedGradientBasedAlgorithms()

    @property
    def Global(self) -> GlobalGradientBasedAlgorithms:
        return GlobalGradientBasedAlgorithms()

    @property
    def LeastSquares(self) -> GradientBasedLeastSquaresAlgorithms:
        return GradientBasedLeastSquaresAlgorithms()

    @property
    def Likelihood(self) -> GradientBasedLikelihoodAlgorithms:
        return GradientBasedLikelihoodAlgorithms()

    @property
    def Local(self) -> GradientBasedLocalAlgorithms:
        return GradientBasedLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> GradientBasedNonlinearConstrainedAlgorithms:
        return GradientBasedNonlinearConstrainedAlgorithms()


@dataclass(frozen=True)
class GradientFreeAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeAlgorithms:
        return BoundedGradientFreeAlgorithms()

    @property
    def Global(self) -> GlobalGradientFreeAlgorithms:
        return GlobalGradientFreeAlgorithms()

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresAlgorithms:
        return GradientFreeLeastSquaresAlgorithms()

    @property
    def Local(self) -> GradientFreeLocalAlgorithms:
        return GradientFreeLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> GradientFreeNonlinearConstrainedAlgorithms:
        return GradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GradientFreeParallelAlgorithms:
        return GradientFreeParallelAlgorithms()


@dataclass(frozen=True)
class GlobalAlgorithms(AlgoSelection):
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalAlgorithms:
        return BoundedGlobalAlgorithms()

    @property
    def GradientBased(self) -> GlobalGradientBasedAlgorithms:
        return GlobalGradientBasedAlgorithms()

    @property
    def GradientFree(self) -> GlobalGradientFreeAlgorithms:
        return GlobalGradientFreeAlgorithms()

    @property
    def NonlinearConstrained(self) -> GlobalNonlinearConstrainedAlgorithms:
        return GlobalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> GlobalParallelAlgorithms:
        return GlobalParallelAlgorithms()


@dataclass(frozen=True)
class LocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLocalAlgorithms:
        return BoundedLocalAlgorithms()

    @property
    def GradientBased(self) -> GradientBasedLocalAlgorithms:
        return GradientBasedLocalAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLocalAlgorithms:
        return GradientFreeLocalAlgorithms()

    @property
    def LeastSquares(self) -> LeastSquaresLocalAlgorithms:
        return LeastSquaresLocalAlgorithms()

    @property
    def Likelihood(self) -> LikelihoodLocalAlgorithms:
        return LikelihoodLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> LocalNonlinearConstrainedAlgorithms:
        return LocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> LocalParallelAlgorithms:
        return LocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalAlgorithms:
        return BoundedGlobalAlgorithms()

    @property
    def GradientBased(self) -> BoundedGradientBasedAlgorithms:
        return BoundedGradientBasedAlgorithms()

    @property
    def GradientFree(self) -> BoundedGradientFreeAlgorithms:
        return BoundedGradientFreeAlgorithms()

    @property
    def LeastSquares(self) -> BoundedLeastSquaresAlgorithms:
        return BoundedLeastSquaresAlgorithms()

    @property
    def Local(self) -> BoundedLocalAlgorithms:
        return BoundedLocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> BoundedNonlinearConstrainedAlgorithms:
        return BoundedNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> BoundedParallelAlgorithms:
        return BoundedParallelAlgorithms()


@dataclass(frozen=True)
class NonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedNonlinearConstrainedAlgorithms:
        return BoundedNonlinearConstrainedAlgorithms()

    @property
    def Global(self) -> GlobalNonlinearConstrainedAlgorithms:
        return GlobalNonlinearConstrainedAlgorithms()

    @property
    def GradientBased(self) -> GradientBasedNonlinearConstrainedAlgorithms:
        return GradientBasedNonlinearConstrainedAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeNonlinearConstrainedAlgorithms:
        return GradientFreeNonlinearConstrainedAlgorithms()

    @property
    def Local(self) -> LocalNonlinearConstrainedAlgorithms:
        return LocalNonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> NonlinearConstrainedParallelAlgorithms:
        return NonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class LeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresAlgorithms:
        return BoundedLeastSquaresAlgorithms()

    @property
    def GradientBased(self) -> GradientBasedLeastSquaresAlgorithms:
        return GradientBasedLeastSquaresAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresAlgorithms:
        return GradientFreeLeastSquaresAlgorithms()

    @property
    def Local(self) -> LeastSquaresLocalAlgorithms:
        return LeastSquaresLocalAlgorithms()

    @property
    def Parallel(self) -> LeastSquaresParallelAlgorithms:
        return LeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class LikelihoodAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH

    @property
    def GradientBased(self) -> GradientBasedLikelihoodAlgorithms:
        return GradientBasedLikelihoodAlgorithms()

    @property
    def Local(self) -> LikelihoodLocalAlgorithms:
        return LikelihoodLocalAlgorithms()


@dataclass(frozen=True)
class ParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    pounders: Type[Pounders] = Pounders
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedParallelAlgorithms:
        return BoundedParallelAlgorithms()

    @property
    def Global(self) -> GlobalParallelAlgorithms:
        return GlobalParallelAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeParallelAlgorithms:
        return GradientFreeParallelAlgorithms()

    @property
    def LeastSquares(self) -> LeastSquaresParallelAlgorithms:
        return LeastSquaresParallelAlgorithms()

    @property
    def Local(self) -> LocalParallelAlgorithms:
        return LocalParallelAlgorithms()

    @property
    def NonlinearConstrained(self) -> NonlinearConstrainedParallelAlgorithms:
        return NonlinearConstrainedParallelAlgorithms()


@dataclass(frozen=True)
class Algorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedAlgorithms:
        return BoundedAlgorithms()

    @property
    def Global(self) -> GlobalAlgorithms:
        return GlobalAlgorithms()

    @property
    def GradientBased(self) -> GradientBasedAlgorithms:
        return GradientBasedAlgorithms()

    @property
    def GradientFree(self) -> GradientFreeAlgorithms:
        return GradientFreeAlgorithms()

    @property
    def LeastSquares(self) -> LeastSquaresAlgorithms:
        return LeastSquaresAlgorithms()

    @property
    def Likelihood(self) -> LikelihoodAlgorithms:
        return LikelihoodAlgorithms()

    @property
    def Local(self) -> LocalAlgorithms:
        return LocalAlgorithms()

    @property
    def NonlinearConstrained(self) -> NonlinearConstrainedAlgorithms:
        return NonlinearConstrainedAlgorithms()

    @property
    def Parallel(self) -> ParallelAlgorithms:
        return ParallelAlgorithms()
