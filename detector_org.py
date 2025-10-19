import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
from typing import Iterable, Optional, Tuple, Union

from .datadTyping import Vector, Num, Array
from .datastore import colors
from .hkl import HKLOrHKLs
from .rotation import rotate_by_axis_angle
from .simu1d import Curve1D
from .simu2d import Pattern2D
from .utils import cross, dot, from_tthgam, isparallel, \
    isperpendicular, tthgam, unit


class Detector:
    """
    Planar detector.

    There are some coordinates systems which are easy to get confused with
    each others and here we are going to clarify them.

    A coordinates system is established against the planar detector. Assume
    that the photosensitive surface of detector is the front side and
    the other side is the back side. The plane of detector can be treated as
    a 2D coordinates system in a 3D space. The x- and y-axis coincide with the
    two edge of rectangle detector detector, and z-axis is perpendicular to
    the plane. Note that
    the coordinates system should be right-handed. So the z-axis is set to
    point to the front size of detector and usually to the sample. The x- and
    y-axis always adapt to the alignment of pixel matrix. A common geometry
    is that the x-axis points to the right horizontally and y-axis points up
    vertically. The x- and y-axis is different from the image coordinates,
    in which x-axis points to the right horizontally and y-axis points down
    vertically. This will be carefully dealed at detector image.

    The unit of length in Detector is arbitary. It can be micrometer,
    millimeter or centimeter. Note keeping the unit of length identical for
    different physical quantity.

    Parameters
    ----------
    normal: vector
        Normal direction of detector in sample coordinates system
        The normal direction is perpendicular to the detector plane and
        points from the backside of detector to the photosensitive side.
        The normal direction usually points to the sample in sample
        diffraction geometry.
    sizex: number
        Size of detector along x-axis. Unit is arbitrary.
    sizey: number
        Size of detector along y-axis. Unit is arbitrary.
    dist: number
        Distance from detector front surface to the sample.
    ponix: number
        Coordinate of PONI along the x-axis of detector.
    poniy: number
        Cooridnate of PONI along the y-axis of detector.
    vx: vector, opitional
        x-axis of detector in sample coordinate
        system. They should be perpendicular to the normal direction.
        The x-axis is perpendicular to the normal direction and
        lies on the xy-plane of experiemental coordinates system default
        and the y-axis can be derived from x-axis and normal direction.
        Not all of these x- and y-axis need given together, one of them can
        be derived from the other by the normal direction.
    vy: vector, optional
        y-axis of detector in sample coordinate system.
    ps: number, optional
        Pixel size.
    """
    def __init__(
            self,
            *,
            normal: Vector,
            sizex: Num,
            sizey: Num,
            dist: Num,
            ponix: Num,
            poniy: Num,
            vx: Vector = None,
            vy: Vector = None,
            ps: Num = None) -> "Detector":
        """
        initialization.
        """
        self._normal = unit(normal)
        self._size = (sizex, sizey)
        self._dist = dist
        self._poni = (ponix, poniy)

        if vx is None and vy is None:
            # default x and y
            parallel = isparallel(self._normal, (0, 0, 1))
            if parallel == 1:
                # cis-parallel (0, 0, 1)
                vvx, vvy = (1, 0, 0), (0, 1, 0)
            elif parallel == -1:
                # trans-parallel (0, 0, -1)
                vvx, vvy = (-1, 0, 0), (0, -1, 0)
            else:
                vvx = unit((normal[1], -normal[0], 0))
                vvy = cross(normal, vx)

        elif vx is None:
            if not isperpendicular(normal, vy):
                raise ValueError(
                    "y is not perpendicular to normal")
            vvy = unit(vy)
            vvx = cross(vy, normal)

        elif vy is None:
            if not isperpendicular(normal, vx):
                raise ValueError(
                    "x is not perpendicular to normal")
            vvx = unit(vx)
            vvy = cross(normal, vx)

        else:
            if not isperpendicular(normal, vx):
                raise ValueError(
                    "x is not perpendicular to normal")
            if not isperpendicular(normal, vy):
                raise ValueError(
                    "y is not perpendicular to normal")
            if not isperpendicular(vx, vy):
                raise ValueError(
                    "x is not perpendicular to y")

            vvx = unit(vx)
            vvy = unit(vy)

        self._x = vvx
        self._y = vvy
        self._ps = ps

    @property
    def size(self) -> Tuple[Num, Num]:
        """
        Size of detector, first one is along x-axis and the
        second one is along y-axis.
        """
        return self._size

    def rotate_by(
            self,
            axis: Vector,
            angle: Num,
            *,
            is_degree: bool = True) -> "Detector":
        """
        Rotate the detector.

        This function only supports axis-angle notation.

        Parameters
        ----------
        axis: vector
            Rotation axis
        angle: number
            Rotation angle
        is_degree: bool, optional
            Whether the angle is in degree
            default True

        Returns
        -------
        detector: Detector
            Rotated detector
        """
        self._normal = rotate_by_axis_angle(
            self._normal, axis=axis, angle=angle, is_degree=is_degree)
        self._x = rotate_by_axis_angle(
            self._x, axis=axis, angle=angle, is_degree=is_degree)
        self._y = rotate_by_axis_angle(
            self._y, axis=axis, angle=angle, is_degree=is_degree)
        return self

    def is_on_by_xy(self, px: Vector, py: Vector) -> Array:
        """
        Determine whether the points are on the detector.

        Parameters
        ----------
        px: vector
            x coordinate
        py: vector
            y coordinate

        Returns
        -------
        array: array of bool
            Whether the points are on the detector.
        """
        not_nan = np.where(np.invert(np.isnan(px)))
        ppx = px[not_nan]
        ppy = py[not_nan]
        cond0 = ppx >= 0
        cond1 = ppx <= self._size[0]
        cond2 = ppy >= 0
        cond3 = ppy <= self._size[1]
        is_on = np.bitwise_and(
            np.bitwise_and(cond0, cond1),
            np.bitwise_and(cond2, cond3)
        )
        res = np.zeros_like(px, dtype=bool)
        res[not_nan] = is_on
        return res

    def tthgam_to_p(
            self,
            tths: Vector,
            gammas: Vector,
            *,
            is_degree: bool = False,
            inc: Optional[Vector] = None,
            vx: Optional[Vector] = None) -> Tuple[Array, Array]:
        """
        Map :math:`2\\theta - \\gamma` pair to the coordinates on
        detector plane.

        If coordinates cannot be calculated, np.nan will be returned.

        Parameters
        ----------
        tths: vector
            Scattering angles :math:`2\\theta`.
        gammas: vector
            Azimuthal angles :math:`\\gamma`.
        is_degree: bool, optional
            If True, the input angles are in degrees. Otherwise the
            angles are in radians.
            The default value is False.
        inc: vector, optional
            Incident direction in sample coordinates system.
            The default direciton is the reverse normal direction of
            this detector.
        vx: vector, optional
            Transverse direction with zero azimuthal angle in sample
            coordinates system.
            The default direction is the x-axis of the detector.

        Returns
        -------
        p: a tuple of two array
            Tuple of two lists containing x coordinate and y coordinate,
            respectively.
        """
        if inc is None:
            inc = -1 * self._normal

        if vx is None:
            vx = self._vx

        n = self._normal
        v = from_tthgam(tths, gammas, inc=inc, vx=vx, is_degree=is_degree)
        p = dot(n, v)
        # if p >= 0, the vector is not point to the detector
        # the point can never be on the detector
        possible_on = p < 0

        if np.ndim(v) == 1:
            v = v[None, :]
            p = p[None]

        filtered_v = v[possible_on]
        filtered_p = p[possible_on]
        v_in_detector = self._dist * (
            - filtered_v / filtered_p[:, None])
        px = np.full_like(p, np.nan)
        py = np.full_like(p, np.nan)
        px[possible_on] = dot(v_in_detector, self._x) + self._poni[0]
        py[possible_on] = dot(v_in_detector, self._y) + self._poni[1]
        return (px, py)

    def is_on_by_tthgam(
            self,
            tths: Vector,
            gammas: Vector,
            *,
            is_degree: bool = False,
            inc: Optional[Vector] = None,
            vx: Optional[Vector] = None) -> Array:
        """
        determine whether the point is on the detector by
        :math:`2\\theta - \\gamma` pair.

        Parameters
        ----------
        tths: vector
            Scattering angles :math:`2\\theta`.
        gammas: vector
            Azimuthal angles :math:`\\gamma`.
        is_degree: bool, optional
            If True, the input angles are in degrees. Otherwise the
            angles are in radians.
            The default value is False.
        inc: vector, optional
            Incident direction in sample coordinates system.
            The default direciton is the reverse normal direction of
            this detector.
        vx: vector, optional
            Transverse direction with zero azimuthal angle in sample
            coordinates system.
            The default direction is the x-axis of the detector.

        Returns
        -------
        is_on: array of bool
            Whether the points are on detector.
        """
        px, py = self.tthgam_to_p(
            tths=tths, gammas=gammas, is_degree=is_degree,
            inc=inc, vx=vx)
        return self.is_on_by_xy(px, py)

    def calc_tthgam_map(
            self,
            *,
            inc: Optional[Vector] = None,
            vx: Optional[Vector] = None):
        """
        Calculate the :math:`2\\theta - \\gamma` pair map on the detector.

        Parameters
        ----------
        inc: vector, optional
            Incident direction in sample coordinates system.
            The default direciton is the reverse normal direction of
            this detector.
        vx: vector, optional
            Transverse direction with zero azimuthal angle in sample
            coordinates system.
            The default direction is the x-axis of the detector.
        """
        if inc is None:
            inc = -1 * self._normal
        if vx is None:
            vx = self._x

        def tthgam_map(
                tths: Vector,
                gammas: Vector,
                *,
                on: bool = True,
                is_degree: bool = False) -> Array:
            """
            map tthgam to px, py.
            If the px and py cannot be calculated, np.nan
            will be returned.

            Parameters
            ----------
            tths: vector
                the scattering angles
            gammas: vector
                the azimuthal angles
            on: bool, optional
                if True, the (px, py) of the tthgams which
                is on the plane but outside the detector will
                be np.nan.
            is_degree: bool, optional
                whether the input tths and gammas are in degrees
                default False
            """
            px, py = self.tthgam_to_p(
                tths, gammas,
                inc=inc, vx=vx,
                is_degree=is_degree
            )
            if on:
                is_on = self.is_on_by_xy(px, py)
                return (
                    np.where(is_on, px, np.nan),
                    np.where(is_on, py, np.nan),
                )
            return px, py

        self._map = tthgam_map
        # poni for point of normal incidence
        self._direct = self._map([0], [0], on=False)

    def p_to_vectors(
            self,
            px: Vector,
            py: Vector) -> Array:
        """
        Map the coordinates in detector coordinates system to the vectors
        in sample coordinates system

        Parameters
        ----------
        px: vectors
            Coordinate along the x-axis of the detector.
        py: vectors
            Coordinate along the y-axis of the detector.

        Returns
        -------
        vectors: an array of vectors
            Vectors in sample coordinate system
        """
        p0 = np.array(px) - self._poni[0]
        p1 = np.array(py) - self._poni[1]

        vs = - self._dist * self._normal \
            + p0[:, None] * self._x \
            + p1[:, None] * self._y

        return vs

    def p_to_tthgam(
            self,
            px: Vector,
            py: Vector,
            *,
            inc: Vector = (0, 0, -1),
            vx: Vector = (1, 0, 0),
            is_degree: bool = False) -> Tuple[Array, Array]:
        """
        Map positions on cylindrical detector to :math:`2\\theta - \\gamma`
        pair.

        Parameters
        ----------
        px: vectors
            Position along the x-axis of the detector.
        py: vectors
            Position along the y-axis of the detector.
        inc: vector, optional
            Incident direction.
            The default direction is the -z-axis of the sample coordinate
            system.
        vx: vector, optional
            Transverse direction with zero azimuthal angle.
            The default direction is the x-axis of the sampel coordinate
            system.

        Returns
        -------
        tthgam: tuple
            :math:`2\\theta - \\gamma` pairs
        """
        return tthgam(
            self.p_to_vectors(px, py),
            inc=inc,
            vx=vx,
            is_degree=is_degree)

    def project_peaks(
            self,
            simu: Union[Curve1D, Pattern2D],
            *,
            inc: Vector = (0, 0, -1),
            vx: Vector = (1, 0, 0),
            **kwargs) -> None:
        """
        project the diffraction simulation result to the detector.

        Parameters
        ----------
        simu: Profile1D or Pattern2D
            Diffraction simulation result
        """
        self._simu = simu

        if isinstance(simu, Curve1D):
            self._inc = inc
            self._vx = vx
            self.calc_tthgam_map(inc=inc, vx=vx)
            self._patterndim = 1
            self.project_peaks_1d(simu, **kwargs)
        elif isinstance(simu, Pattern2D):
            self._inc = simu.inc
            self._vx = simu.vx
            self.calc_tthgam_map(inc=simu.inc, vx=simu.vx)
            self._patterndim = 2
            self.project_peaks_2d(simu, **kwargs)

    def project_peaks_1d(
            self,
            simu1d: Curve1D,
            *,
            num_gammas: int = 2000) -> None:
        """
        Project the 1D diffraction curve onto the detector.

        Parameters
        ----------
        simu1d: Curve1D
            1D diffraction curve.
        num_gammas: int, optional
            Number of gammas increments.
            The default number of increments is 2000.
        """
        gammas = np.linspace(0, 2 * np.pi, num_gammas)
        self._px = []
        self._py = []
        self._intns = []
        intns = simu1d.peak_intensities()
        mask = np.ones(simu1d.peak_num(), dtype=bool)
        for i, tth in enumerate(simu1d.peak_angles(is_degree=False)):
            tths = tth * np.ones(num_gammas)
            px, py = self._map(tths, gammas, on=False, is_degree=False)
            is_on = np.any(self.is_on_by_xy(px, py))
            if not is_on:
                mask[i] = False
            else:
                self._px.append(px)
                self._py.append(py)
                self._intns.append(np.full_like(px, intns[i]))

        self._px = np.array(self._px)
        self._py = np.array(self._py)
        self._intns = np.array(self._intns)
        self._mask = mask
        factor = self.geometry_factor(self._px, self._py)
        self._intns *= factor

    def project_peaks_2d(
            self,
            simu2d: Pattern2D) -> None:
        """
        Project 2D diffraction pattern onto the detector.
        """
        px, py = self._map(
            simu2d.peak_tths(is_degree=False),
            simu2d.peak_gammas(is_degree=False),
            on=True)
        mask = np.invert(np.isnan(px))
        self._px = px[mask]
        self._py = py[mask]
        self._intns = simu2d.peak_intensities()[mask]
        self._mask = mask
        factor = self.geometry_factor(self._px, self._py)
        self._intns *= factor

    def geometry_factor(self, px: Vector, py: Vector) -> Array:
        """
        Geometry factor
        """
        shape = np.shape(px)
        px = np.ravel(px)
        py = np.ravel(py)
        xToPedal = px - self._poni[0]
        yToPedal = py - self._poni[1]
        distance2Pedal = np.hypot(xToPedal, yToPedal)
        s2p = np.hypot(distance2Pedal, self._dist)
        factor = (self._dist / s2p) ** 3
        factor = np.reshape(factor, shape)
        return factor

    def peak_found(self, hkls: HKLOrHKLs) -> Union[bool, Iterable[bool]]:
        """
        Whether the peaks with given Miller index triplets are on the detector.
        """
        hkls = np.array(hkls)
        shape = hkls.shape
        peak_hkls = self._simu.peak_hkls()[self._mask]

        hkls = np.expand_dims(hkls, axis=-1)
        peak_hkls = np.expand_dims(np.transpose(peak_hkls), axis=0)

        # shrink the hkl axis
        res = np.all(hkls == peak_hkls, axis=-2)

        # shrink the peak_hkls axis
        res = np.any(res, axis=-1)
        return res.reshape(shape[:-1])

    def is_peaks_on(self) -> bool:
        """
        Whether any peaks are projected onto the detector
        """
        mask = self._mask
        return np.any(mask)

    def show(
            self,
            *,
            ax=None,
            show_label: bool = False,
            show_color: bool = True,
            show_direct: bool = True,
            show_poni: bool = True,
            **kwargs):
        """
        Plot to show the pattern on detector.

        Parameters
        ----------
        ax: matplotlib axes, optional
            Axis to be plotted in.
            If not specified, a new axis will be created.
        show_label: bool, optional
            Whether plotting the label.
            default False.
        show_color: bool, optional
            Whether colorizing the points.
            default True.
        show_direct: bool, optional
            Whether plotting the direct beam.
            default True.
        show_poni: bool, optional
            Whether plotting the PONI.
            default True.
        kwargs:
            all other keyword arguments will be passed
            to `axis.scatter()` during the plotting.
        """
        if not hasattr(self, '_patterndim'):
            raise ValueError("Not pattern!")

        if self._patterndim == 1:
            self.show_1d(
                ax=ax,
                show_label=show_label,
                show_color=show_color,
                show_poni=show_poni,
                **kwargs)
        elif self._patterndim == 2:
            self.show_2d(
                ax=ax,
                show_label=show_label,
                show_color=show_color,
                show_poni=show_poni,
                **kwargs)

    def show_2d(
            self,
            *,
            ax=None,
            show_label: bool = False,
            show_color: bool = True,
            show_direct: bool = True,
            show_poni: bool = True,
            **kwargs) -> None:
        """
        plot 2d pattern
        """
        def hklshow(hkl):
            return "[{:d} {:d} {:d}]".format(
                hkl[0], hkl[1], hkl[2])

        mask = self._mask
        if not np.any(mask):
            print("No peaks is on the detector")
            return

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            ax.set(
                xlabel=r'x',
                ylabel=r'y',
                title='Pattern on detector')

        tths = self._simu.peak_tths()[mask]
        gammas = self._simu.peak_gammas()[mask]
        intns = self._intns
        hkls = self._simu.peak_hkls()[mask]
        px = self._px
        py = self._py

        # group the spots with same position
        order = np.lexsort((py, px))
        px = px[order]
        py = py[order]
        hkls = hkls[order, :]
        tths = tths[order]
        gammas = gammas[order]
        intns = intns[order]
        is_overlap = np.bitwise_and(
            np.isclose(np.diff(px), 0),
            np.isclose(np.diff(py), 0),
        )

        represent_indices = []
        represent_intensities = []
        represent_children = []

        index = 0
        all_intn = intns[0]
        this_indices = [0]
        for i, overlap in enumerate(is_overlap):
            if overlap:
                all_intn += intns[i + 1]
                this_indices.append(i + 1)
                if intns[i + 1] > intns[index]:
                    index = i + 1
            else:
                represent_indices.append(index)
                represent_intensities.append(all_intn)
                represent_children.append(this_indices)
                index = i + 1
                all_intn = intns[i + 1]
                this_indices = [i + 1, ]

        # add last indices
        represent_indices.append(index)
        represent_intensities.append(all_intn)
        represent_children.append(this_indices)
        represent_px = px[represent_indices]
        represent_py = py[represent_indices]
        represent_hkls = hkls[represent_indices]

        cmap = plt.get_cmap("jet")
        legends = []
        if show_direct and hasattr(self, "_direct"):
            legend = ax.scatter(
                self._poni[0],
                self._poni[1],
                s=100,
                marker="s",
                c="tab:orange",
                label="direct"
            )
            legends.append(legend)

        if show_poni and hasattr(self, "_poni"):
            poni = [self._poni[0], self._poni[1]]
            legend = ax.scatter(
                poni[0],
                poni[1],
                s=100,
                marker="*",
                c='r',
                label="PONI"
            )
            legends.append(legend)

        if show_color:
            pcm = ax.scatter(
                represent_px,
                represent_py,
                s=10,
                marker="x",
                c=represent_intensities,
                cmap=cmap,
                **kwargs)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(pcm, cax=cax)
        else:
            ax.scatter(px, py, marker="x", s=20, **kwargs)

        if show_label:
            for i, hkl in enumerate(represent_hkls):
                ax.annotate(
                    hklshow(hkl),
                    xy=(represent_px[i], represent_py[i]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    size=8)

        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        if len(legends) > 0:
            plt.legend(
                handles=legends,
                loc="lower right",
                bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()

    def show_1d(
            self,
            *,
            ax=None,
            show_label: bool = False,
            show_color: bool = True,
            show_direct: bool = True,
            show_poni: bool = True,
            **kwargs) -> None:
        """
        show 1d profile in detector
        """
        def hklshow(hkl):
            return "{{{:d} {:d} {:d}}}".format(
                hkl[0], hkl[1], hkl[2]
            )

        mask = self._mask
        if not np.any(mask):
            print("No peaks is on the detector")
            return

        hkls = self._simu.peak_hkls()[mask]
        intns = self._intns
        grains = self._simu.peak_grains()[mask]
        px = self._px
        py = self._py
        num = np.count_nonzero(mask)

        if ax is None:
            fig = plt.figure()
            fig.subplots_adjust(right=0.7)
            ax = fig.gca()
            ax.set_aspect(1)
            ax.set(
                xlabel=r'x',
                ylabel=r'y',
                title='Pattern on detector')

        if show_direct and hasattr(self, "_direct"):
            ax.scatter(
                self._direct[0],
                self._direct[1],
                s=100,
                marker="s",
                c="tab:orange",
                label="direct"
            )

        if show_poni and hasattr(self, "_poni"):
            poni = [self._poni[0], self._poni[1]]
            ax.scatter(
                poni[0],
                poni[1],
                s=100,
                marker="*",
                c='r',
                label="PONI"
            )

        baseSize = 5
        size = intns / np.nanmax(intns) * baseSize
        len_colors = len(colors)
        for i in range(num):
            ppx, ppy = px[i], py[i]
            ssize = size[i]

            # print lines with variable linewidth
            points = np.array([ppx, ppy]).T.reshape(-1, 1, 2)
            lw = ssize[:-1]
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, lw=lw,
                label="{} {}".format(grains[i].name, hklshow(hkls[i])),
                colors=colors[i % len_colors])
            ax.add_collection(lc)

        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.legend(loc=2, bbox_to_anchor=(1.05, 1))

        # plt.tight_layout()
        plt.show()

    def init_pic(self, ps: Optional[Num] = None) -> None:
        """
        Initialize a picture with the same size of detector

        Note that the y-axis of the picture is the reverse y-axis of
        detector.

        Parameters
        ----------
        ps: number, optional
            Pixel size of the picture.
            default the pixel size set in this detector
            If the pixel size is not set in detector and this ps is also
            not set, a ValueError will be raised

        """
        if ps is None:
            if self._ps is None:
                raise ValueError("the pixel size is not set")
        else:
            self._ps = ps

        # the size in picture is different from the size in common sense
        # For a 2D array, the first axis is x-axis and the second axis is
        # y-axis as ones expect. However in the 2D array representation of
        # image, the first axis is y-axis and the second axis is x-axis.
        # so we have to exchange the size of x-axis and y-axis with each
        # other
        self._resolution = np.ceil(
            (self._size[1] / ps, self._size[0] / ps),
        ).astype(int)
        self.pic_intensity = np.zeros(self._resolution, dtype=np.float)

        pixel_x = self._ps * np.arange(self._resolution[1])
        pixel_y = self._ps * np.arange(self._resolution[0])
        pixel_grid_x, pixel_grid_y = np.meshgrid(pixel_x, pixel_y)
        factors = self.geometry_factor(pixel_grid_x, pixel_grid_y)
        pixel_grid_x = pixel_grid_x.ravel()
        pixel_grid_y = pixel_grid_y.ravel()
        tth_map, gamma_map = self.p_to_tthgam(
            pixel_grid_x, pixel_grid_y, inc=self._inc, vx=self._vx
        )

        self._pixel_x = pixel_x
        self._pixel_y = pixel_y
        self._tth_map = tth_map.reshape(self._resolution)
        self._gamma_map = gamma_map.reshape(self._resolution)
        self._factors = factors

    def calc_to_pic(
            self,
            ps: Num,
            show: bool = True,
            *args, **kwargs) -> None:
        """
        Calculate the intensities of each pixels.

        Parameters
        ----------
        ps: number, optional
            Pixel size
        show: bool, optional
            If plotting the calculated picture.
            The default value is True.
        """
        self.init_pic(ps=ps)

        if self._patterndim == 1:
            self.calc_to_pic_1d(*args, **kwargs)
        elif self._patterndim == 2:
            self.calc_to_pic_2d(*args, **kwargs)

        if show:
            fig = plt.figure()
            ax = fig.gca()
            # in the 2d array representation of image， the first axis
            # here transpose the pic and lower the origin to
            # coincide with the detector show results
            pcm = ax.imshow(self.pic, cmap='jet', origin="lower")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(pcm, cax=cax)
            plt.tight_layout()
            plt.show()

    def calc_to_pic_1d(
            self,
            tth_res: Num = 0.1,
            *,
            is_degree: bool = True):
        """
        Calculate the intensities of each pixels after a 1D diffraction curve
        is projected.

        Parameters
        ----------
        tth_res: number, optional
            Resolution of :math:`2\\theta`.
            The default resolution is 0.1 degree.
        is_degree: bool, optional
            If True, the input angles are in degrees, otherwise in radians.
            The default value is True.
        """
        if is_degree:
            tth_res = np.deg2rad(tth_res)
        tth_map = self._tth_map
        mask = self._mask
        tths = self._simu.peak_angles(is_degree=False)[mask]
        intns = self._simu.peak_intensities()[mask]
        for tth, intensity in zip(tths, intns):
            is_in = np.isclose(tth_map, tth, atol=tth_res)
            self.pic_intensity[is_in] += intensity

        self.pic_intensity *= self._factors
        self.change_pic_intensity_to_pic()

    def calc_to_pic_2d(self,
                       *,
                       sigmax: int = 1,
                       sigmay: int = 1,
                       rot_angle: Num = 0,
                       is_degree: bool = True):
        """
        Calculate the intensities of each pixels after a 2D diffraction pattern
        is projected.

        Parameters
        ----------
        sigmax: int
            Sigma (standard deviation) value of the Gaussian peak
            broadening along the first main axis.
            Unit is pixel.
            The default value is 1.
        sigmay: int
            Sigma (standard deviation) value of the Gaussian peak
            broadening along the second main axis.
            Unit is pixel.
            The default value is 1.
        rot_angle: number
            Angle between the main axes and the detector axes.
            The default angle is 0 degree.
        is_degree: bool, optional
            If True, the input angles are in degrees, otherwise in radians.
            The default value is True.
        """
        px = np.ravel(self._px)
        py = np.ravel(self._py)
        intns = np.ravel(self._intns)
        is_nan = np.any(
            (np.isnan(px), np.isnan(py), np.isnan(intns)),
            axis=0
        )
        order = np.where(np.invert(is_nan))
        px = px[order]
        py = py[order]
        intns = intns[order]

        index_x, index_y = self.calc_index(px, py)

        peak_seed_x, peak_seed_y, peak_seed_i = _peak_seed(
            sigmax, sigmay, rot_angle, is_degree=is_degree)

        for ix, iy, i in zip(index_x, index_y, intns):
            ppx = ix + peak_seed_x
            ppy = iy + peak_seed_y
            intensity = i * peak_seed_i
            inside = np.all(
                (
                    ppx >= 0, ppx < self._resolution[1],
                    ppy >= 0, ppy < self._resolution[0]
                ),
                axis=0
            )
            ppx = ppx[inside]
            ppy = ppy[inside]
            intensity = intensity[inside]
            self.pic_intensity[ppy, ppx] += intensity

        self.change_pic_intensity_to_pic()

    def change_pic_intensity_to_pic(self):
        intn_max = self.pic_intensity.max()
        pic = self.pic_intensity / intn_max
        self.pic = (65535 * pic).astype(np.uint16)

    def calc_index(self, px: Vector, py: Vector) -> Tuple[Vector, Vector]:
        index_x = (px / self._ps).astype(int)
        index_y = (py / self._ps).astype(int)
        return index_x, index_y

    def save_pic(self, filepath: str):
        """
        Save the picture to file at given path.
        """
        if not hasattr(self, "pic"):
            raise ValueError("pic is not calculated")

        m = Image.fromarray(self.pic)
        m.save(filepath)


def _peak_seed(
        sigmax: Num, sigmay: Num,
        rot_angle: Num = 0,
        *, is_degree: bool = True) -> Tuple[Vector, Vector, Vector]:
    """
    returns the seed intensity distribution
    """
    if sigmax == 0 or sigmay == 0:
        return np.array([0, ]), np.array([0, ]), np.array([1, ])

    # outside the cutoff, intensities are considered zero
    cut_off_x = 3 * sigmax
    cut_off_y = 3 * sigmay

    p_x = np.arange(-cut_off_x, cut_off_x + 1, dtype=int)
    p_y = np.arange(-cut_off_y, cut_off_y + 1, dtype=int)
    p_y, p_x = np.meshgrid(p_y, p_x)
    p_x = np.concatenate(p_x)
    p_y = np.concatenate(p_y)

    if is_degree:
        rot_angle = np.deg2rad(rot_angle)

    # f(x,y) = A \exp(-(a(x - x_0)^2 + 2b(x - x_0)(y - y_0) + c(y - y_0)^2))
    # rot_angle is the angle between the main axis of gaussian peaks and the
    # global axis
    a = np.cos(rot_angle) ** 2 / (2 * sigmax * sigmax) + \
        np.sin(rot_angle) ** 2 / (2 * sigmay * sigmay)

    b = - np.sin(2 * rot_angle) / (4 * sigmax * sigmax) + \
        np.sin(2 * rot_angle) / (4 * sigmay * sigmay)

    c = np.sin(rot_angle) ** 2 / (2 * sigmax * sigmax) + \
        np.cos(rot_angle) ** 2 / (2 * sigmay * sigmay)

    intensity = np.exp(- (a * p_x * p_x + 2 * b * p_x * p_y + c * p_y * p_y))

    return p_x, p_y, intensity
