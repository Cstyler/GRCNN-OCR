from typing import Dict, List, MutableMapping, Union

SymbolType = Dict[str, Union[Dict[str, Dict[str, str]], str]]
SymbolsType = List[SymbolType]
WordType = Dict[str, SymbolsType]
WordsType = List[WordType]
BlockType = Dict[str, List[Dict[str, WordsType]]]
BlocksType = List[BlockType]
FullTextAnnotationType = Dict[str, List[Dict[str, BlocksType]]]
TextAnnotationsType = List[Dict[str, str]]
ResponseType = Dict[str, Union[TextAnnotationsType, FullTextAnnotationType]]
BarCodeType = Dict[str, Union[str, bool]]
SegmentType = Dict[str, Union[float, int, str]]
SegsType = Dict[str, SegmentType]
MarketsDictType = Dict[str, Dict[str, Dict[str, SegsType]]]
MetricDictType = Dict[str, Dict[str, Union[float, int]]]

ConfigType = MutableMapping[str, MutableMapping[str, str]]

