import numpy as np
from ezmsg.util.messages.axisarray import (
    AxisArray,
    slice_along_axis,
    replace,
)
import ezmsg.core as ez

from .base import BaseSignalTransformer, BaseSignalTransformerUnit


class TransposeSettings(ez.Settings):
    """
    Settings for :obj:`Transpose` node.

    Fields:
      axes:
    """

    axes: tuple[int | str | type(...), ...] | None = None
    order: str | None = None


class TransposeState(ez.State):
    axes_ints: tuple[int, ...] | None = None
    hash: int = 0


class TransposeTransformer(
    BaseSignalTransformer[TransposeState, TransposeSettings, AxisArray]
):
    """
    Downsampled data simply comprise every `factor`th sample.
    This should only be used following appropriate lowpass filtering.
    If your pipeline does not already have lowpass filtering then consider
    using the :obj:`Decimate` collection instead.
    """

    def _hash_message(self, message: AxisArray) -> int:
        return hash(tuple(message.dims))

    def check_metadata(self, message: AxisArray) -> bool:
        return self.state.hash != self._hash_message(message)

    def reset(self, message: AxisArray) -> None:
        self._state.hash = self._hash_message(message)

        if self.settings.axes is None:
            self._state.axes_ints = None
        else:
            ell_ix = [ix for ix, ax in enumerate(self.settings.axes) if ax is Ellipsis]
            if len(ell_ix) > 1:
                raise ValueError("Only one Ellipsis is allowed in axes.")
            ell_ix = ell_ix[0] if len(ell_ix) == 1 else len(message.dims)
            prefix = []
            for ax in self.settings.axes[:ell_ix]:
                if isinstance(ax, int):
                    prefix.append(ax)
                else:
                    if ax not in message.dims:
                        raise ValueError(f"Axis {ax} not found in message dims.")
                    prefix.append(message.dims.index(ax))
            suffix = []
            for ax in self.settings.axes[ell_ix + 1 :]:
                if isinstance(ax, int):
                    suffix.append(ax)
                else:
                    if ax not in message.dims:
                        raise ValueError(f"Axis {ax} not found in message dims.")
                    suffix.append(message.dims.index(ax))
            ells = [
                _
                for _ in range(message.data.ndim)
                if _ not in prefix and _ not in suffix
            ]
            re_ix = tuple(prefix + ells + suffix)
            if re_ix == tuple(range(message.data.ndim)):
                self._state.axes_ints = None
            else:
                self._state.axes_ints = re_ix
        if self.settings.order is not None and self.settings.order.upper()[0] not in [
            "C",
            "F",
        ]:
            raise ValueError("order must be 'C' or 'F'.")

    def __call__(self, message: AxisArray) -> AxisArray:
        if self.settings.axes is None and self.settings.order is None:
            # Passthrough
            return message
        return super().__call__(message)

    def _process(self, message: AxisArray) -> AxisArray:
        if self.state.axes_ints is None:
            # No transpose required
            if self.settings.order is None:
                # No memory relayout required
                # Note: We should not be able to reach here because it should be shortcutted at passthrough.
                msg_out = message
            else:
                # If the memory is already contiguous in the correct order, np.require won't do anything.
                msg_out = replace(
                    message,
                    data=np.require(
                        message.data, requirements=self.settings.order.upper()[0]
                    ),
                )
        else:
            dims_out = [message.dims[ix] for ix in self.state.axes_ints]
            data_out = np.transpose(message.data, axes=self.state.axes_ints)
            if self.settings.order is not None:
                data_out = np.require(
                    data_out, requirements=self.settings.order.upper()[0]
                )
            msg_out = replace(
                message,
                data=data_out,
                dims=dims_out,
            )
        return msg_out


def transpose(
    axes: tuple[int | str | type(...), ...] | None = None, order: str | None = None
) -> TransposeTransformer:
    return TransposeTransformer(TransposeSettings(axes=axes, order=order))


class Transpose(
    BaseSignalTransformerUnit[
        TransposeState, TransposeSettings, AxisArray, TransposeTransformer
    ]
):
    SETTINGS = TransposeSettings
