module class4_tree20(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[376] & (~i[355])) | ((~i[376]) & (~i[355])) | ((~i[167]) & (~i[91]) & i[355]) | (i[248] & (~i[395]) & i[91] & i[355]) | ((~i[248]) & (~i[395]) & i[91] & i[355]) | (i[124] & i[186] & i[395] & i[91] & i[355]) | ((~i[124]) & i[186] & i[395] & i[91] & i[355]) | (i[190] & (~i[186]) & i[395] & i[91] & i[355]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
