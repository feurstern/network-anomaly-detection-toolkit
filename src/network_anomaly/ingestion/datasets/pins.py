"""Pinned download locations and checksums of the supported public datasets.

Generated from verified downloads on 2026-10-30. GitHub-hosted files are pinned to a commit;
each file has its SHA-256 where it was computed, otherwise its git blob id at that commit (the
content hash git itself uses), which the downloader verifies the same way.
"""

NAB_COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
SMD_COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"

# path in the repository -> (size in bytes, sha256)
NAB_FILES: dict[str, tuple[int, str]] = {
    "LICENSE.txt": (1056, "0a0b4d0b10cb1f7ed9ab2993ef93defc03447e6eba9daca1315dd32dae4877e3"),
    "data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv": (
        105367,
        "ab446fbd8b9f37507eb2fdb06315826d8daeef02e241133ce06e0ee571ba53d9",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv": (
        116041,
        "8942e498de7b40f1b4a6d802755592c09b8fb73ce8f658763a09cb9ea1ea94ac",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv": (
        119229,
        "01613e6f632d067f11a5dfd40a188b0789752b388d9bc77a398bd06333878a76",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv": (
        106007,
        "90ceabd570b449241ee24ff8a116a793979b7671ce4490707311bfea0e0aae1f",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv": (
        113675,
        "d768419037c9db269343822957314f57ee21a7d9a4d41df2add0d1ba45ba84de",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv": (
        116980,
        "749a15c2e1a4543c21fee9cbf3338cd8a7ed5f5f8a1308b9b099b06c2c66e66b",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv": (
        104962,
        "d936cea74682ed43ac96b778352d7de294c0b0c168f4a7e6817162346cdb28c1",
    ),
    "data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv": (
        115788,
        "f3433f8171f4dcea86c0c7af9996d0f166f812fa0f4567f1d5cd85d2d2cd69b4",
    ),
    "data/realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv": (
        116626,
        "ce3d6c6a652ed7c31a6dd3727a36e128fcac6fb2381c63267ad0fbc02eccd9df",
    ),
    "data/realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv": (
        101626,
        "a0e734a66098bb81833835f0839dadafbc8ae829c51494670ee8849e46eb5b63",
    ),
    "data/realAWSCloudwatch/ec2_network_in_257a54.csv": (
        117236,
        "39104b08f2e0a673b5137eb7681897fcadf0955fedf565740a6a94edc63a81a4",
    ),
    "data/realAWSCloudwatch/ec2_network_in_5abac7.csv": (
        120827,
        "c27ef93f582af63bf42e80d804764cbe64ae987620f3cfab5477d6ebb3c6157e",
    ),
    "data/realAWSCloudwatch/elb_request_count_8c0756.csv": (
        101052,
        "74c26574a01ca9fb89dddb5021e2e13c3a93eb25dc640438a9acb1ceb00f1021",
    ),
    "data/realAWSCloudwatch/grok_asg_anomaly.csv": (
        131065,
        "86a0abe9d58e376858d2fd4f4a438f1997e24c64a09e5af03accd4e1b372ac47",
    ),
    "data/realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv": (
        37339,
        "f115fbe6542159eed5612445bfa620708a28ab6b41349d1432bb15c3a3b8414d",
    ),
    "data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv": (
        117806,
        "d5df979ef85928769a016cd0f935919d8c2a113926e21e1eb3a5001d688760e2",
    ),
    "data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv": (
        113599,
        "6b712b922ab3b3404c5a629d64a570c390c99471ee8b9b0ff7665f553fda92c1",
    ),
    "data/realKnownCause/cpu_utilization_asg_misconfiguration.csv": (
        539274,
        "58ba65dc0737cfbac11b51514476d50c438d44011232144bb8d93f392df58f9f",
    ),
    "data/realKnownCause/ec2_request_latency_system_failure.csv": (
        123428,
        "98378580aa80157e057c61d59d81daddccc6c65a2c0c800e3f01f603b8215c3f",
    ),
    "labels/combined_labels.json": (
        7015,
        "98724014c162fcfd4be605683517f194051a54e98756c53667bd33ddbc644d9b",
    ),
    "labels/combined_windows.json": (
        15359,
        "1e1fbc4601321aad8d0f8b3784c8134299379f68f6c1f7777565f8ffd57ab6b1",
    ),
}

# path in the repository -> (size in bytes, sha256 or None, git blob sha1)
SMD_FILES: dict[str, tuple[int, str | None, str]] = {
    "ServerMachineDataset/LICENSE": (
        1072,
        "c9bef1aab5be4ae6e0735a9f5974de043b8a1894bd8a2a7a163e36027b8895df",
        "a35f9e98caf3d5d6d6164ef9a303ec4216780c01",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-1.txt": (
        311,
        "ea2b75dacb9cd140d81962203b18706e8e85d70c87dcced69746a37ba133fa48",
        "d61a908e062bd383f968dabe416ff521269cb220",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-2.txt": (
        294,
        None,
        "3de06f16510cf1c8d41f2972b56ed506b50469e7",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-3.txt": (
        352,
        None,
        "d91dde2f34ba28bbf85fd17e403fa937f7eae16e",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-4.txt": (
        421,
        None,
        "1f1b1cb1fb6f978ee60486eb493b15cbd8b3f0d7",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-5.txt": (
        276,
        None,
        "e9a8b7190bb538cf6bbed56f387e9b02d95709e4",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-6.txt": (
        1127,
        None,
        "b5f6c5f48b1ece06b47a508ed528735e0a0ce553",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-7.txt": (
        499,
        None,
        "1422cf5a40c3ea314d8d18862a7d89ce2a23322f",
    ),
    "ServerMachineDataset/interpretation_label/machine-1-8.txt": (
        494,
        None,
        "9a570cd70069deb2ef8e8f1a3eb040108afefa69",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-1.txt": (
        389,
        None,
        "e1dcecdd2c53646ff41b34c5b541cc5cebb38bb3",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-2.txt": (
        313,
        None,
        "5d32f19de76e85456606c2124df4c8ea2022dc20",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-3.txt": (
        329,
        None,
        "189c869ce8241a05ae8face51380804f27336601",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-4.txt": (
        460,
        None,
        "2bbe352b2042d0dc6fa312cdf095a94e80c5e57c",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-5.txt": (
        513,
        None,
        "7a3dcac3f25506cd340099683cfacf31ee524cd9",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-6.txt": (
        320,
        None,
        "737bd2fbdcb9628af3acc9f81f7ec66685597923",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-7.txt": (
        457,
        None,
        "388f91825a82e59511cc97172a982c3084fcebc9",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-8.txt": (
        106,
        None,
        "fca07cc58ff17dc327fcdfcbcd31da2b27242659",
    ),
    "ServerMachineDataset/interpretation_label/machine-2-9.txt": (
        376,
        None,
        "5ef64ed19b06c57cfd79c7a952a6977eed783a29",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-1.txt": (
        151,
        None,
        "10f44107441d76b0ae524bd311cbfece62512d99",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-10.txt": (
        416,
        None,
        "d0746beb2a227a927b3df8c265a15326947495f5",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-11.txt": (
        159,
        None,
        "cfc667a65e0136341baff9dc7d995cfc7850fc89",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-2.txt": (
        188,
        None,
        "3176eea5fa50af3b2661f7c6b32f3580ead66fc7",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-3.txt": (
        785,
        None,
        "e29ee5f4aa05f89e742fb94aca43325f9410267c",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-4.txt": (
        166,
        None,
        "1161482ce070a054fb0e7f2a252cd1af6476e290",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-5.txt": (
        254,
        None,
        "fbd87c033da75285c9f2513bd4b6558106e656b6",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-6.txt": (
        379,
        None,
        "afabf91e5c27d4a198037cc4b8d99c720f7dc413",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-7.txt": (
        247,
        None,
        "3544dd60d76181271e4bb6ae6c06022f7f91f4be",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-8.txt": (
        226,
        None,
        "8c72b2c5164cf0170e62a99b9910132bbd9e2f86",
    ),
    "ServerMachineDataset/interpretation_label/machine-3-9.txt": (
        166,
        None,
        "0af845563eff0f0d49c299543ca7fad48f526dcc",
    ),
    "ServerMachineDataset/test/machine-1-1.txt": (
        9739818,
        "bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb",
        "0c48ff71d3d72128731b7045f49a33d9383a969e",
    ),
    "ServerMachineDataset/test/machine-1-2.txt": (
        8103348,
        None,
        "86e8b1a03db0aa31edd8b95130fd79ba6693619c",
    ),
    "ServerMachineDataset/test/machine-1-3.txt": (
        8106426,
        None,
        "620621931063671396a635b296b9eed93aa77e9e",
    ),
    "ServerMachineDataset/test/machine-1-4.txt": (
        8107794,
        None,
        "4ce08f550455ae6e6e383f85c8f93246d3f7a789",
    ),
    "ServerMachineDataset/test/machine-1-5.txt": (
        8107452,
        None,
        "093b8d44af05b4d3a92ac81bc0e1bda5a8ccca35",
    ),
    "ServerMachineDataset/test/machine-1-6.txt": (
        8101638,
        None,
        "345df3517fa084e52c8958709a3cb68ff0f376fb",
    ),
    "ServerMachineDataset/test/machine-1-7.txt": (
        8104374,
        None,
        "f717524daf4167710f1cacf81aeaed43b220fabe",
    ),
    "ServerMachineDataset/test/machine-1-8.txt": (
        8105058,
        None,
        "b29fdf6ce95c8ae47cfd8c061fd4bb5a57a8f781",
    ),
    "ServerMachineDataset/test/machine-2-1.txt": (
        8103348,
        None,
        "f6d71b1f1595d2333dd963b39e4c54d5812cb7c8",
    ),
    "ServerMachineDataset/test/machine-2-2.txt": (
        8105400,
        None,
        "45867eb9cff6cf0f434ed9f4b3981dedbe923b16",
    ),
    "ServerMachineDataset/test/machine-2-3.txt": (
        8101638,
        None,
        "e6b14abd219100263093255532d8d13db5a53bad",
    ),
    "ServerMachineDataset/test/machine-2-4.txt": (
        8101638,
        None,
        "72895129d2a243d78d64a25fc1a9d1ef6d66e3b2",
    ),
    "ServerMachineDataset/test/machine-2-5.txt": (
        8101638,
        None,
        "774b3bb9e7603a4eb992dafd3a3b30ec07e75b43",
    ),
    "ServerMachineDataset/test/machine-2-6.txt": (
        9830106,
        None,
        "a7973287dcd76c15ef068ab63fbb95156559ff11",
    ),
    "ServerMachineDataset/test/machine-2-7.txt": (
        8104032,
        None,
        "db7058cc2e31c0f456e2b6aa0b0bc206e724e7bc",
    ),
    "ServerMachineDataset/test/machine-2-8.txt": (
        8106426,
        None,
        "72ebea6c02c13112d10001f91d5e153774d4e766",
    ),
    "ServerMachineDataset/test/machine-2-9.txt": (
        9822924,
        None,
        "853289b07c1067c4ae998592655dd292d644f50d",
    ),
    "ServerMachineDataset/test/machine-3-1.txt": (
        9815400,
        None,
        "936b6ddf1b981f049dca104c135b5e057fa41f7d",
    ),
    "ServerMachineDataset/test/machine-3-10.txt": (
        8103006,
        None,
        "7191328c304bd94cf16433b6cfa24334752a5a33",
    ),
    "ServerMachineDataset/test/machine-3-11.txt": (
        9814032,
        None,
        "5f51ad22b02c859025936b8dc19cf43755edd238",
    ),
    "ServerMachineDataset/test/machine-3-2.txt": (
        8106426,
        None,
        "fbedbc03660c4207ad3f52748bf850252b8a8cf8",
    ),
    "ServerMachineDataset/test/machine-3-3.txt": (
        8106426,
        None,
        "d41089fc31be0c609f736a1913f485c807dfc69e",
    ),
    "ServerMachineDataset/test/machine-3-4.txt": (
        8100954,
        None,
        "994417b8a350a406452ae0a45bc7ae91f69a0651",
    ),
    "ServerMachineDataset/test/machine-3-5.txt": (
        8102322,
        None,
        "b46ef99fb34a08613b2a50412b7242bb6d78e72e",
    ),
    "ServerMachineDataset/test/machine-3-6.txt": (
        9824292,
        None,
        "d074b9c2bad837079fc2eda5356a6a4d0ad6dde3",
    ),
    "ServerMachineDataset/test/machine-3-7.txt": (
        9817110,
        None,
        "4deaedf4f43b736d5fc13f0b4b7d50863a71fdd4",
    ),
    "ServerMachineDataset/test/machine-3-8.txt": (
        9816768,
        None,
        "1fc673c2e62c047739db649e88c91efb045b10ca",
    ),
    "ServerMachineDataset/test/machine-3-9.txt": (
        9819846,
        None,
        "b76365fa344c435f64ce54259882186b0b10be72",
    ),
    "ServerMachineDataset/test_label/machine-1-1.txt": (
        56958,
        "cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19",
        "2b1063d42ba0fa3db0fd9296e4686cb4bca50260",
    ),
    "ServerMachineDataset/test_label/machine-1-2.txt": (
        47388,
        None,
        "29bdb081f95cf546ba6d499302fbdc4f6137de79",
    ),
    "ServerMachineDataset/test_label/machine-1-3.txt": (
        47406,
        None,
        "2eed09ad2e98125d25f2700918d86796ed22b0c4",
    ),
    "ServerMachineDataset/test_label/machine-1-4.txt": (
        47414,
        None,
        "b1745658b4a2efcd2b3c6af1f260b1a27a5a7ff3",
    ),
    "ServerMachineDataset/test_label/machine-1-5.txt": (
        47412,
        None,
        "2ba2cb279b1a6f813743db93af4f1fe832caf8b2",
    ),
    "ServerMachineDataset/test_label/machine-1-6.txt": (
        47378,
        None,
        "bff3b67fc10d2b7c89c778f746390852b2c24bc8",
    ),
    "ServerMachineDataset/test_label/machine-1-7.txt": (
        47394,
        None,
        "120964780cc40a4c68ff1b68ddb7258094b564e9",
    ),
    "ServerMachineDataset/test_label/machine-1-8.txt": (
        47398,
        None,
        "40eea4944b436383e255f2eeb7255fac0f0edac4",
    ),
    "ServerMachineDataset/test_label/machine-2-1.txt": (
        47388,
        None,
        "d072491a7f5d4fc5d6854e5d9cb782352834adcf",
    ),
    "ServerMachineDataset/test_label/machine-2-2.txt": (
        47400,
        None,
        "1cedfe43e4c6e25f6f416b2af78c6a95d0e63b2f",
    ),
    "ServerMachineDataset/test_label/machine-2-3.txt": (
        47378,
        None,
        "60880105c01c983ef1158f1cf7f4e2feb0607e98",
    ),
    "ServerMachineDataset/test_label/machine-2-4.txt": (
        47378,
        None,
        "eb0965c3ff1e5c0168950dc51e5050ffdeb18c32",
    ),
    "ServerMachineDataset/test_label/machine-2-5.txt": (
        47378,
        None,
        "18e0cf48f655bc8021f2222c928a411d69c59eae",
    ),
    "ServerMachineDataset/test_label/machine-2-6.txt": (
        57486,
        None,
        "bcd73d41f984013358d1fbba9ef6dfd0bc74bbf7",
    ),
    "ServerMachineDataset/test_label/machine-2-7.txt": (
        47392,
        None,
        "2384b0bcdd87afa2b3b2d430a29062da66f33c51",
    ),
    "ServerMachineDataset/test_label/machine-2-8.txt": (
        47406,
        None,
        "43edaf5e053c966a0aa520eb7e0d20ba6dabbd30",
    ),
    "ServerMachineDataset/test_label/machine-2-9.txt": (
        57444,
        None,
        "7e7bf05e3943348231d940b8f1654e147240f921",
    ),
    "ServerMachineDataset/test_label/machine-3-1.txt": (
        57400,
        None,
        "46242ec08583cddd5f08dcfe3191cef3f5a88624",
    ),
    "ServerMachineDataset/test_label/machine-3-10.txt": (
        47386,
        None,
        "accb1de8b01fe9199e428de06ba78382f4a18c0b",
    ),
    "ServerMachineDataset/test_label/machine-3-11.txt": (
        57392,
        None,
        "c027c2f0e38da95ddbd49a0c1d2a49ffc67bc251",
    ),
    "ServerMachineDataset/test_label/machine-3-2.txt": (
        47406,
        None,
        "3bfe413eb37c82db4a823c49da9f35105ada73b6",
    ),
    "ServerMachineDataset/test_label/machine-3-3.txt": (
        47406,
        None,
        "ef7b8f1126c1e4d94dfa4f3879f262081b441ddb",
    ),
    "ServerMachineDataset/test_label/machine-3-4.txt": (
        47374,
        None,
        "71ccfd12e7bc504320f7fbe6cd9974d1df93368d",
    ),
    "ServerMachineDataset/test_label/machine-3-5.txt": (
        47382,
        None,
        "d2b038344a723186fcdc788ea67faf1dc5e7e145",
    ),
    "ServerMachineDataset/test_label/machine-3-6.txt": (
        57452,
        None,
        "98c30e746dc8b08c4c3883fc84396a2c06f3541b",
    ),
    "ServerMachineDataset/test_label/machine-3-7.txt": (
        57410,
        None,
        "34e2fa46ab80de19e65a02d016fdae28583a0351",
    ),
    "ServerMachineDataset/test_label/machine-3-8.txt": (
        57408,
        None,
        "f175565bd6b285e1b3cce05d685280ed166324ad",
    ),
    "ServerMachineDataset/test_label/machine-3-9.txt": (
        57426,
        None,
        "69f83c4d58b4410473ae9c6904c81e87da669b82",
    ),
    "ServerMachineDataset/train/machine-1-1.txt": (
        9739818,
        "ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753",
        "01303e82d3033210d1bfd6272199fc42f2cf82d1",
    ),
    "ServerMachineDataset/train/machine-1-2.txt": (
        8103348,
        None,
        "5027d3a80976a974afc4597a8e7402354a3d14a7",
    ),
    "ServerMachineDataset/train/machine-1-3.txt": (
        8106084,
        None,
        "253862b50af3ba881aac527e4cc2a56e32e25509",
    ),
    "ServerMachineDataset/train/machine-1-4.txt": (
        8107452,
        None,
        "e05a441ace8bb76b4949fe8930e809f5b1ebd51a",
    ),
    "ServerMachineDataset/train/machine-1-5.txt": (
        8107110,
        None,
        "6f916154de35a59f2b03afca1fcd096d14c2ebb7",
    ),
    "ServerMachineDataset/train/machine-1-6.txt": (
        8101296,
        None,
        "52c895f51cd4cac94b3800c98cda8b5bf5ceaede",
    ),
    "ServerMachineDataset/train/machine-1-7.txt": (
        8104374,
        None,
        "b0f9440d017fb64821604d2efca1300ebf507161",
    ),
    "ServerMachineDataset/train/machine-1-8.txt": (
        8104716,
        None,
        "a38a45e1ae0f6e1f7d8ad4c3de9aca9f21d530c4",
    ),
    "ServerMachineDataset/train/machine-2-1.txt": (
        8103006,
        None,
        "85e6986e5955ad1137ee0492ab663c9998ab8c2b",
    ),
    "ServerMachineDataset/train/machine-2-2.txt": (
        8105058,
        None,
        "28275bdc7f3ced6550e78d840489869548c8d315",
    ),
    "ServerMachineDataset/train/machine-2-3.txt": (
        8101296,
        None,
        "c82505805b7c9166bf52734dea07a6557ce1ffba",
    ),
    "ServerMachineDataset/train/machine-2-4.txt": (
        8101638,
        None,
        "52d3c0c19b4de102e9c0724236cc6b2e064fcbc0",
    ),
    "ServerMachineDataset/train/machine-2-5.txt": (
        8101296,
        None,
        "cb59861e3f1164470cb1f5dce8ca05a869a8e7dd",
    ),
    "ServerMachineDataset/train/machine-2-6.txt": (
        9830106,
        None,
        "0cef966acf151b747c40c15a902b81bd6710cde5",
    ),
    "ServerMachineDataset/train/machine-2-7.txt": (
        8104032,
        None,
        "c695bfc4b42d688680d0825b577a97809c7c1831",
    ),
    "ServerMachineDataset/train/machine-2-8.txt": (
        8106084,
        None,
        "dc8c8fa5f78f9037108c806e45b155eea7ec493a",
    ),
    "ServerMachineDataset/train/machine-2-9.txt": (
        9822924,
        None,
        "ae01edd03cf87342fdb9d768c78e423f4b239747",
    ),
    "ServerMachineDataset/train/machine-3-1.txt": (
        9815400,
        None,
        "e2e5d6b3382370e36b2336dcde0d388a125eb016",
    ),
    "ServerMachineDataset/train/machine-3-10.txt": (
        8102664,
        None,
        "7fd68b1d1cfb6619b5426f760ae53b9b1273dfa1",
    ),
    "ServerMachineDataset/train/machine-3-11.txt": (
        9813690,
        None,
        "335ca33f5670a73acef4da557cb3f32974a38ea9",
    ),
    "ServerMachineDataset/train/machine-3-2.txt": (
        8106084,
        None,
        "0159c59a689e376a08c120248579579d1d2799c2",
    ),
    "ServerMachineDataset/train/machine-3-3.txt": (
        8106426,
        None,
        "aaf865412f625af2a91f2dfd388ffd513393fb05",
    ),
    "ServerMachineDataset/train/machine-3-4.txt": (
        8100954,
        None,
        "a62c27c9396e64b7bff50ca09af2be46d33439a9",
    ),
    "ServerMachineDataset/train/machine-3-5.txt": (
        8101980,
        None,
        "376264a15daa939208680dff40299d3807c9dbfe",
    ),
    "ServerMachineDataset/train/machine-3-6.txt": (
        9824292,
        None,
        "bd86ae5dd7c9393b56a4b614f95232c79ccfc299",
    ),
    "ServerMachineDataset/train/machine-3-7.txt": (
        9817110,
        None,
        "af5f4b0c9f67306e312a94e49eb19149d7ff45f8",
    ),
    "ServerMachineDataset/train/machine-3-8.txt": (
        9816426,
        None,
        "200305b1ea1e87d45fb0568f7d73bf1c76100200",
    ),
    "ServerMachineDataset/train/machine-3-9.txt": (
        9819846,
        None,
        "11713d81ed1d844a6930af2a4c41c5154587f74c",
    ),
}

# Single archives: (url, size in bytes, sha256)
CICIDS2017_IMPROVED = (
    "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip",
    343549013,
    "97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883",
)
CIDDS_001 = (
    "https://www.hs-coburg.de/wp-content/uploads/2024/11/CIDDS-001.zip",
    402679606,
    "5145d13b2c4ee278390743964f242759ffbf2823336dc7361b4d7255a04c54d9",
)
