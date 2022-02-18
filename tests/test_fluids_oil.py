import pytest

from digirock import OilPVT, OilBW92, DeadOil, BaseOil


# @pytest.fixture()
# def mock_deadoil(request: SubRequest, test_data):
#     param = getattr(request, "param", None)
#     test = DeadOil(api=35)

#     pvt = param[:-1][0]
#     ans = param[-1]

#     if pvt == "const":
#         test.set_fvf(1.4)
#     elif pvt == "calc":
#         test.calc_fvf(120)
#     elif pvt == "text":
#         pvt = np.loadtxt(test_data / "PVT_BO.inc")
#         test.set_fvf(pvt[:, 1], pvt[:, 0])
#     else:
#         raise ValueError(f"Unknown pvt key {pvt} in fixture")
#     yield test, ans


# class TestDeadOil:
#     def test_init(self):
#         assert isinstance(DeadOil(), DeadOil)

#     @pytest.mark.parametrize(
#         "api, rho, ans",
#         [
#             (None, None, (None, None)),
#             (30, None, (30, 0.8761609)),
#             (None, 0.6, (104.333333, 0.6)),
#         ],
#     )
#     def test_init_kw(self, api, rho, ans):
#         test = Oil(api=api, std_density=rho)
#         try:
#             assert (test.api, test.std_density) == approx(ans)
#         except TypeError:
#             assert (test.api, test.std_density) == ans

#     @pytest.mark.parametrize(
#         "mock_deadoil",
#         [
#             ("const", float),
#             ("calc", float),
#             ("text", xr.DataArray),
#         ],
#         indirect=True,
#     )
#     def test_set_fvf(self, mock_deadoil):
#         MockDeadOil, ans = mock_deadoil
#         assert isinstance(MockDeadOil.bo, ans)

#     @pytest.mark.parametrize(
#         "mock_deadoil",
#         [
#             ("const", float),
#             ("calc", float),
#         ],
#         indirect=True,
#     )
#     def test_fvf(self, mock_deadoil):
#         MockDeadOil, ans = mock_deadoil
#         assert isinstance(MockDeadOil.fvf(), ans)

#     @pytest.mark.parametrize(
#         "pres, fvf",
#         [
#             (0, 1.08810526),
#             (100, 1.248),
#             (200, 1.398),
#             (300, 1.467),
#             (500, 1.427),
#             (np.r_[100, 200], np.r_[1.248, 1.398]),
#         ],
#     )
#     @pytest.mark.parametrize("mock_deadoil", [("text", None)], indirect=True)
#     def test_fvf_table(self, mock_deadoil, pres, fvf):
#         MockDeadOil, ans = mock_deadoil
#         assert MockDeadOil.fvf(pres) == approx(fvf)

#     @pytest.mark.parametrize("mock_deadoil", [("calc", 1.095994)], indirect=True)
#     def test_calc_fvf(self, mock_deadoil):
#         MockDeadOil, ans = mock_deadoil
#         assert MockDeadOil.fvf() == approx(ans)

#     # def test_density(self, temp, pres):

#     # def test_velocity(self, temp, pres):

#     # def test_modulus(self, temp, pres):


# @pytest.fixture()
# def mock_oil(request: SubRequest, test_data):
#     param = getattr(request, "param", None)
#     test = Oil(api=35)

#     pvt = param[:-1][0]
#     ans = param[-1]

#     if pvt == "const":
#         test.set_disolved_gas(0.9, 100)
#         test.set_fvf(1.4, 100)
#     elif pvt == "calc":
#         test.set_disolved_gas(0.9, 100)
#         test.calc_fvf(120, 300)
#     elif pvt == "text":
#         pvt = np.loadtxt(test_data / "PVT_BO.inc")
#         test.set_disolved_gas(0.9, 100)
#         test.set_fvf(pvt[:, 1], 100, pvt[:, 0])
#     elif pvt == "text2":
#         pvt = np.loadtxt(test_data / "PVT_BO.inc")
#         test.set_disolved_gas(0.9, 100)
#         test.set_fvf(pvt[:, 1], 100, pvt[:, 0])
#         test.set_fvf(pvt[:, 1], 120, pvt[:, 0])
#     elif pvt == "pvto":
#         test.set_disolved_gas(0.9, 100)
#         test.load_pvto(test_data / "COMPLEX_PVT.inc")
#     else:
#         raise ValueError(f"Unknown pvt key {pvt} in fixture")
#     yield test, ans


# class TestOil:
#     def test_init(self):
#         assert isinstance(Oil(), Oil)

#     @pytest.mark.parametrize(
#         "api, rho, ans",
#         [
#             (None, None, (None, None)),
#             (30, None, (30, 0.8761609)),
#             (None, 0.6, (104.333333, 0.6)),
#         ],
#     )
#     def test_init_kw(self, api, rho, ans):
#         test = Oil(api=api, std_density=rho)
#         try:
#             assert (test.api, test.std_density) == approx(ans)
#         except TypeError:
#             assert (test.api, test.std_density) == ans

#     @pytest.mark.parametrize(
#         "mock_oil", [("text", xr.DataArray), ("text2", xr.DataArray)], indirect=True
#     )
#     def test_set_fvf(self, mock_oil):
#         mockOil, ans = mock_oil
#         assert isinstance(mockOil.bo, ans)

#     @pytest.mark.parametrize("mock_oil", [("pvto", xr.DataArray)], indirect=True)
#     def test_load_pvto(self, mock_oil):
#         mockOil, ans = mock_oil
#         assert isinstance(mockOil.bo, ans)

# @pytest.mark.parametrize(
#     "mock_oil", [("const", float), ("calc", float)], indirect=True
# )
# def test_calc_fvf_type(self, mock_oil):
#     mockOil, ans = mock_oil
#     assert isinstance(mockOil.fvf(100), ans)

# @pytest.mark.parametrize(
#     "mock_oil",
#     [("const", 1.4), ("calc", 1.386388), ("text", 1.467), ("pvto", 1.34504596)],
#     indirect=True,
# )
# def test_calc_fvf(self, mock_oil):
#     mockOil, ans = mock_oil
#     assert mockOil.fvf(300) == approx(ans)

# def test_
